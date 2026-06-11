"""
TradeMind AI — Pydantic Response Schemas (Pydantic v2)

All response_model declarations live here so Swagger shows
the exact keys and types returned by every endpoint.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict

AnyDate = Optional[Union[datetime, str]]   # DB may return datetime or ISO string


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class StatusOut(BaseModel):
    status: str


class StatusMessageOut(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Auth / User
# ---------------------------------------------------------------------------

class UserOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    virtual_balance: float = 1_000_000.0
    virtual_invested: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    mode: Optional[str] = "PAPER"
    created_at: AnyDate = None


class AuthOut(BaseModel):
    status: str
    user: UserOut
    token: str


class UserCreateOut(BaseModel):
    status: str
    user: UserOut


# ---------------------------------------------------------------------------
# Trade Signals — nested sub-models
# ---------------------------------------------------------------------------

class TradeInfoOut(BaseModel):
    type: Optional[str] = None
    buy_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    expected_return_pct: Optional[float] = None


class PriceInfoOut(BaseModel):
    current: Optional[float] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None


class ModelInfoOut(BaseModel):
    name: Optional[str] = None
    horizon: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None


class PositionSizingOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    avg_daily_volume: Optional[int] = None
    daily_turnover_cr: Optional[float] = None
    liquidity: Optional[str] = None
    max_safe_qty: Optional[int] = None
    max_qty_per_user: Optional[int] = None
    max_investment_per_user: Optional[float] = None
    min_qty: Optional[int] = None
    volatility_pct: Optional[float] = None
    delivery_pct: Optional[float] = None


class TopDriverOut(BaseModel):
    feature: str
    importance: float


class TradeSignalOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    name: Optional[str] = None
    signal: str                          # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
    confidence: float
    trade: Optional[TradeInfoOut] = None
    price: Optional[PriceInfoOut] = None
    model: Optional[ModelInfoOut] = None
    position: Optional[PositionSizingOut] = None
    sentiment: Optional[Dict[str, float]] = None
    top_drivers: Optional[List[TopDriverOut]] = None
    generated_at: AnyDate = None


class SignalSummaryOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    STRONG_BUY: Optional[int] = 0
    BUY: Optional[int] = 0
    HOLD: Optional[int] = 0
    SELL: Optional[int] = 0
    STRONG_SELL: Optional[int] = 0


class PaginatedSignalsOut(BaseModel):
    data: List[TradeSignalOut]
    total: int
    page: int
    size: int
    summary: Optional[SignalSummaryOut] = None


class SignalForStockOut(BaseModel):
    signal: Optional[TradeSignalOut] = None


class SignalHistoryOut(BaseModel):
    data: List[Any]
    total: int


# ---------------------------------------------------------------------------
# Positions & Orders
# ---------------------------------------------------------------------------

class PositionOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    user_id: Optional[int] = None
    symbol: str
    name: Optional[str] = None
    quantity: Optional[int] = None
    buy_price: Optional[float] = None
    current_price: Optional[float] = None
    invested: Optional[float] = None
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    signal: Optional[str] = None
    confidence: Optional[float] = None
    horizon: Optional[str] = None
    mode: Optional[str] = None
    created_at: AnyDate = None
    updated_at: AnyDate = None


class PaginatedPositionsOut(BaseModel):
    user_id: int
    data: List[PositionOut]
    positions: List[PositionOut]
    total: int
    page: int
    size: int
    count: int


class OrderOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    user_id: Optional[int] = None
    symbol: str
    name: Optional[str] = None
    order_type: Optional[str] = None       # BUY / SELL
    order_purpose: Optional[str] = None    # ENTRY / STOP_LOSS / TARGET / SQUARE_OFF
    quantity: Optional[int] = None
    price: Optional[float] = None
    value: Optional[float] = None
    status: Optional[str] = None
    pnl: Optional[float] = None
    signal: Optional[str] = None
    confidence: Optional[float] = None
    horizon: Optional[str] = None
    mode: Optional[str] = None
    gtt_rule_id: Optional[str] = None
    bracket_id: Optional[str] = None
    created_at: AnyDate = None


class PaginatedOrdersOut(BaseModel):
    user_id: int
    data: List[OrderOut]
    orders: List[OrderOut]
    total: int
    page: int
    size: int
    count: int


# ---------------------------------------------------------------------------
# Execute Signal
# ---------------------------------------------------------------------------

class GttInfoOut(BaseModel):
    sl_rule_id: Optional[str] = None
    target_rule_id: Optional[str] = None


class ExecutePositionOut(BaseModel):
    symbol: str
    name: Optional[str] = None
    quantity: int
    buy_price: float
    invested: float
    target: float
    stop_loss: float
    fees: Optional[float] = None


class ExecuteAccountOut(BaseModel):
    balance_before: float
    balance_after: float
    total_invested: float


class RiskCheckOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    passed: bool
    message: Optional[str] = None


class ExecuteSignalOut(BaseModel):
    status: str
    bracket_id: Optional[str] = None
    mode: Optional[str] = None
    orders: Optional[List[OrderOut]] = None
    gtt: Optional[GttInfoOut] = None
    position: Optional[ExecutePositionOut] = None
    account: Optional[ExecuteAccountOut] = None
    risk_checks: Optional[List[RiskCheckOut]] = None
    reason: Optional[str] = None           # present when status == "rejected"


class SquareOffAccountOut(BaseModel):
    balance: float
    total_invested: float
    total_pnl: float
    win_count: int
    loss_count: int


class SquareOffOut(BaseModel):
    status: str
    symbol: Optional[str] = None
    quantity: Optional[int] = None
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    invested: Optional[float] = None
    sell_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: Optional[float] = None
    result: Optional[str] = None
    account: Optional[SquareOffAccountOut] = None


class SquareOffAllOut(BaseModel):
    status: str
    positions_closed: int
    results: List[Any]
    account: Optional[SquareOffAccountOut] = None


# ---------------------------------------------------------------------------
# Portfolio Summary
# ---------------------------------------------------------------------------

class PortfolioUserOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None


class PortfolioSummaryOut(BaseModel):
    user: PortfolioUserOut
    balance: float
    invested: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    open_positions: int
    wins: int
    losses: int
    win_rate: float
    positions: List[PositionOut]


# ---------------------------------------------------------------------------
# Risk Settings
# ---------------------------------------------------------------------------

class RiskSettingsOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    user_id: Optional[int] = None
    max_daily_loss: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_position_pct: Optional[float] = None
    auto_stop_loss: Optional[int] = None
    auto_target: Optional[int] = None
    updated_at: AnyDate = None


class RiskSettingsUpdateOut(BaseModel):
    status: str
    settings: RiskSettingsOut


# ---------------------------------------------------------------------------
# Today P&L
# ---------------------------------------------------------------------------

class TodayPnlOut(BaseModel):
    date: str
    profit: float
    loss: float
    net_pnl: float
    today_pnl: float
    today_pnl_pct: float
    trades_closed: int


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class NotificationOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    user_id: Optional[int] = None
    type: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None
    message: Optional[str] = None
    created_at: AnyDate = None
    is_read: Optional[bool] = False


class NotificationsListOut(BaseModel):
    data: List[NotificationOut]
    unread: int


# ---------------------------------------------------------------------------
# GTT Orders
# ---------------------------------------------------------------------------

class GTTOrderOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    user_id: Optional[int] = None
    symbol: Optional[str] = None
    gtt_rule_id: Optional[str] = None
    order_type: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    status: Optional[str] = None
    created_at: AnyDate = None


class GTTOrdersOut(BaseModel):
    data: List[GTTOrderOut]
    total: int
    user_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

class WatchlistItemOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    user_id: Optional[int] = None
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None
    added_at: AnyDate = None


class WatchlistOut(BaseModel):
    data: List[WatchlistItemOut]
    total: int


class WatchlistAddOut(BaseModel):
    status: str
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

class StockOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    price: Optional[float] = None        # alias for close
    volume: Optional[int] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    prev_close: Optional[float] = None
    date: AnyDate = None
    # Signal fields — null when no signal exists
    signal: Optional[str] = None
    confidence: Optional[float] = None
    horizon: Optional[str] = None
    expected_return_pct: Optional[float] = None
    expReturn: Optional[float] = None    # alias for frontend Stock type
    sentiment: Optional[float] = None
    updatedMin: Optional[int] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


class PaginatedStocksOut(BaseModel):
    data: List[StockOut]
    total: int
    page: int
    size: int
