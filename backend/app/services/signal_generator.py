"""
Signal Generator Service

Generate trading signals from model predictions.
"""

import logging
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Stock, OHLCData, TechnicalIndicator, Signal

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate and store trading signals."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_version = settings.current_model_version
        self.confidence_threshold = settings.signal_confidence_threshold
    
    def determine_signal_type(
        self,
        prob_buy: float,
        prob_hold: float,
        prob_avoid: float,
    ) -> str:
        """
        Determine signal type from probabilities.
        
        Args:
            prob_buy: Probability of BUY signal
            prob_hold: Probability of HOLD signal
            prob_avoid: Probability of AVOID signal
            
        Returns:
            Signal type: BUY, HOLD, or AVOID
        """
        max_prob = max(prob_buy, prob_hold, prob_avoid)
        
        if max_prob == prob_buy:
            return "BUY"
        elif max_prob == prob_avoid:
            return "AVOID"
        else:
            return "HOLD"
    
    def calculate_confidence(
        self,
        prob_buy: float,
        prob_hold: float,
        prob_avoid: float,
        signal_type: str,
    ) -> float:
        """
        Calculate confidence score for a signal.
        
        Higher confidence when:
        - Primary probability is high
        - Other probabilities are low
        - Clear separation between probabilities
        """
        probs = {"BUY": prob_buy, "HOLD": prob_hold, "AVOID": prob_avoid}
        primary_prob = probs[signal_type]
        
        # Remove primary and get the next highest
        del probs[signal_type]
        secondary_prob = max(probs.values())
        
        # Confidence based on margin
        margin = primary_prob - secondary_prob
        confidence = min(0.95, primary_prob * 0.7 + margin * 0.3)
        
        return round(confidence, 3)
    
    def calculate_risk_reward(
        self,
        current_price: float,
        atr: float,
        signal_type: str,
        confidence: float,
    ) -> tuple[float, Optional[float], Optional[float]]:
        """
        Calculate risk-reward ratio and price targets.
        
        Args:
            current_price: Current closing price
            atr: Average True Range
            signal_type: Signal type
            confidence: Signal confidence
            
        Returns:
            Tuple of (risk_reward_ratio, stop_loss, target)
        """
        if signal_type != "BUY" or atr is None:
            return None, None, None
        
        # Risk: 1.5-2x ATR as stop loss
        risk_multiplier = 1.5 if confidence >= 0.7 else 2.0
        stop_loss = current_price - (atr * risk_multiplier)
        
        # Reward: 2-3x the risk based on confidence
        reward_multiplier = 3.0 if confidence >= 0.7 else 2.0
        target = current_price + (atr * risk_multiplier * reward_multiplier)
        
        risk = current_price - stop_loss
        reward = target - current_price
        
        risk_reward = round(reward / risk, 2) if risk > 0 else None
        
        return risk_reward, round(stop_loss, 2), round(target, 2)
    
    def determine_timeframe(self, confidence: float, volatility: float) -> int:
        """
        Determine suggested holding period in days.
        
        Higher confidence + lower volatility = longer timeframe
        """
        base_days = 10
        
        if confidence >= 0.8:
            base_days = 15
        elif confidence >= 0.6:
            base_days = 10
        else:
            base_days = 5
        
        # Adjust for volatility
        if volatility and volatility > 0.30:
            base_days = int(base_days * 0.7)
        elif volatility and volatility < 0.15:
            base_days = int(base_days * 1.3)
        
        return max(5, min(30, base_days))
    
    def generate_reasoning(
        self,
        signal_type: str,
        confidence: float,
        rsi: Optional[float],
        market_regime: Optional[str],
        returns_20d: Optional[float],
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        reasons = []
        
        if signal_type == "BUY":
            if rsi and rsi < 40:
                reasons.append("RSI indicates oversold conditions")
            if returns_20d and returns_20d < -5:
                reasons.append("Recent pullback may present opportunity")
            if confidence >= 0.75:
                reasons.append("Strong technical alignment")
        elif signal_type == "AVOID":
            if rsi and rsi > 70:
                reasons.append("RSI indicates overbought conditions")
            if market_regime == "BEAR":
                reasons.append("Bearish market regime")
            if returns_20d and returns_20d > 20:
                reasons.append("Extended rally may face resistance")
        else:
            reasons.append("Mixed signals suggest holding current position")
        
        if not reasons:
            reasons.append("Based on technical indicator analysis")
        
        return ". ".join(reasons) + "."
    
    async def generate_signal_for_stock(
        self,
        stock: Stock,
        probabilities: dict,
        signal_date: Optional[date] = None,
    ) -> Optional[Signal]:
        """
        Generate and store signal for a stock.
        
        Args:
            stock: Stock model instance
            probabilities: Dict with buy, hold, avoid probabilities
            signal_date: Date for signal (defaults to today)
            
        Returns:
            Created Signal or None
        """
        if signal_date is None:
            signal_date = date.today()
        
        prob_buy = probabilities.get("buy", 0.33)
        prob_hold = probabilities.get("hold", 0.34)
        prob_avoid = probabilities.get("avoid", 0.33)
        
        # Determine signal
        signal_type = self.determine_signal_type(prob_buy, prob_hold, prob_avoid)
        confidence = self.calculate_confidence(prob_buy, prob_hold, prob_avoid, signal_type)
        
        # Get latest indicators and price
        indicator_result = await self.db.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.stock_id == stock.id)
            .order_by(TechnicalIndicator.date.desc())
            .limit(1)
        )
        indicator = indicator_result.scalar_one_or_none()
        
        ohlc_result = await self.db.execute(
            select(OHLCData)
            .where(OHLCData.stock_id == stock.id)
            .order_by(OHLCData.date.desc())
            .limit(1)
        )
        ohlc = ohlc_result.scalar_one_or_none()
        
        current_price = ohlc.close if ohlc else None
        atr = indicator.atr_14 if indicator else None
        volatility = indicator.volatility_20 if indicator else None
        rsi = indicator.rsi_14 if indicator else None
        market_regime = indicator.market_regime if indicator else None
        returns_20d = indicator.returns_20d if indicator else None
        
        # Calculate risk-reward
        risk_reward, stop_loss, target = self.calculate_risk_reward(
            current_price or 0, atr, signal_type, confidence
        )
        
        # Determine timeframe
        timeframe = self.determine_timeframe(confidence, volatility)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(
            signal_type, confidence, rsi, market_regime, returns_20d
        )
        
        # Check if signal already exists
        existing = await self.db.execute(
            select(Signal).where(
                Signal.stock_id == stock.id,
                Signal.date == signal_date,
                Signal.model_version == self.model_version,
            )
        )
        
        if existing.scalar_one_or_none():
            logger.info(f"Signal already exists for {stock.symbol} on {signal_date}")
            return None
        
        # Create signal
        signal = Signal(
            stock_id=stock.id,
            date=signal_date,
            signal_type=signal_type,
            probability_buy=prob_buy,
            probability_hold=prob_hold,
            probability_avoid=prob_avoid,
            confidence=confidence,
            risk_reward_ratio=risk_reward,
            suggested_timeframe_days=timeframe,
            suggested_entry=current_price,
            suggested_stop_loss=stop_loss,
            suggested_target=target,
            model_version=self.model_version,
            reasoning=reasoning,
            generated_at=datetime.utcnow(),
        )
        
        self.db.add(signal)
        await self.db.commit()
        
        logger.info(
            f"Generated {signal_type} signal for {stock.symbol} "
            f"with {confidence:.1%} confidence"
        )
        
        return signal
    
    async def generate_all_signals(
        self,
        predictions: dict[int, dict],
        signal_date: Optional[date] = None,
    ) -> dict:
        """
        Generate signals for all stocks with predictions.
        
        Args:
            predictions: Dict mapping stock_id to probabilities
            signal_date: Date for signals
            
        Returns:
            Summary of signal generation
        """
        summary = {
            "total": len(predictions),
            "buy_signals": 0,
            "hold_signals": 0,
            "avoid_signals": 0,
            "high_confidence": 0,
            "failed": 0,
        }
        
        for stock_id, probs in predictions.items():
            try:
                stock_result = await self.db.execute(
                    select(Stock).where(Stock.id == stock_id)
                )
                stock = stock_result.scalar_one_or_none()
                
                if not stock:
                    continue
                
                signal = await self.generate_signal_for_stock(stock, probs, signal_date)
                
                if signal:
                    if signal.signal_type == "BUY":
                        summary["buy_signals"] += 1
                    elif signal.signal_type == "HOLD":
                        summary["hold_signals"] += 1
                    else:
                        summary["avoid_signals"] += 1
                    
                    if signal.confidence >= 0.7:
                        summary["high_confidence"] += 1
                        
            except Exception as e:
                logger.error(f"Failed to generate signal for stock {stock_id}: {e}")
                summary["failed"] += 1
        
        return summary
