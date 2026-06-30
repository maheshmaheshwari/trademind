"""
Email template: password reset OTP.

Usage:
    from api.emails.password_reset import render
    content = render(otp="123456", requested_at="Mon, 30 Jun 2026 10:30 UTC", ip_address="1.2.3.4")
    # content["subject"], content["html"], content["text"]
"""

from __future__ import annotations


def render(otp: str, requested_at: str, ip_address: str | None = None) -> dict:
    """Return subject, HTML, and plain-text bodies for the password-reset OTP email."""
    ip_html = (
        f'<p style="margin:0;font-size:13px;color:#0E1726">'
        f'<span style="color:#6B7890">IP Address&nbsp;</span>{ip_address}</p>'
        if ip_address else ""
    )

    ip_text = f"  IP Address: {ip_address}\n" if ip_address else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Password Reset — TradeMind AI</title>
</head>
<body style="margin:0;padding:0;background:#F4F6FB;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#F4F6FB;padding:40px 16px">
    <tr><td align="center">
      <table width="100%" cellpadding="0" cellspacing="0" style="max-width:520px">

        <!-- Header -->
        <tr>
          <td style="background:#0A0E1A;padding:28px 32px;border-radius:14px 14px 0 0">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td>
                  <span style="display:inline-block;background:#2563EB;color:#ffffff;
                               font-weight:800;font-size:13px;padding:5px 11px;
                               border-radius:7px;letter-spacing:.5px">TM</span>
                  <span style="color:#EEF2F9;font-size:18px;font-weight:700;
                               margin-left:10px;vertical-align:middle">TradeMind AI</span>
                </td>
              </tr>
              <tr>
                <td style="padding-top:5px">
                  <span style="color:#6B7890;font-size:12px">AI-Powered Trading Intelligence</span>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="background:#ffffff;padding:36px 32px">

            <h1 style="margin:0 0 8px;font-size:22px;font-weight:700;color:#0E1726">
              Password Reset Request
            </h1>
            <div style="width:40px;height:3px;background:#2563EB;border-radius:2px;margin-bottom:24px"></div>

            <p style="margin:0 0 24px;color:#4A5670;font-size:15px;line-height:1.6">
              We received a request to reset the password for your TradeMind account.
              Use the one-time password below to proceed.
            </p>

            <!-- OTP box -->
            <div style="background:#EEF1F8;border:2px solid #DBEAFE;border-radius:12px;
                        padding:28px 20px;text-align:center;margin-bottom:24px">
              <p style="margin:0 0 14px;font-size:11px;font-weight:700;color:#4A5670;
                        text-transform:uppercase;letter-spacing:2px">
                One-Time Password
              </p>
              <span style="font-size:44px;font-weight:800;letter-spacing:14px;color:#2563EB;
                           font-variant-numeric:tabular-nums">
                {otp}
              </span>
              <p style="margin:18px 0 0;font-size:13px;color:#6B7890">
                &#9201;&nbsp; Valid for <strong style="color:#0E1726">15 minutes</strong> only
              </p>
            </div>

            <!-- Request details -->
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border-radius:10px;margin-bottom:24px;
                          border-left:3px solid #3B82F6;background:#F4F6FB">
              <tr>
                <td style="padding:16px 20px">
                  <p style="margin:0 0 12px;font-size:11px;font-weight:700;color:#4A5670;
                             text-transform:uppercase;letter-spacing:1.5px">
                    Request Details
                  </p>
                  <p style="margin:0 0 6px;font-size:13px;color:#0E1726">
                    <span style="color:#6B7890">Time&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                    {requested_at}
                  </p>
                  {ip_html}
                </td>
              </tr>
            </table>

            <!-- Security warning -->
            <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;
                        padding:16px 20px">
              <p style="margin:0;font-size:13px;color:#92400E;line-height:1.6">
                <strong>&#128274; Didn't request this?</strong><br>
                You can safely ignore this email. Your password will <em>not</em> be
                changed unless you enter this OTP on TradeMind.
              </p>
            </div>

          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#0A0E1A;padding:22px 32px;border-radius:0 0 14px 14px;
                     text-align:center">
            <p style="margin:0 0 6px;font-size:12px;color:#6B7890">
              &copy; 2026 TradeMind AI &nbsp;&middot;&nbsp; All rights reserved
            </p>
            <p style="margin:0;font-size:11px;color:#3D4E6B">
              This is an automated security email. Please do not reply.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text = f"""TradeMind AI — Password Reset
==============================

We received a request to reset the password for your TradeMind account.

Your one-time password (OTP):

  {otp}

This OTP is valid for 15 minutes only.

Request Details:
  Time:       {requested_at}
{ip_text}
Didn't request this? You can safely ignore this email.
Your password will NOT be changed unless you enter this OTP on TradeMind.

-------------------------------
© 2026 TradeMind AI · All rights reserved
This is an automated security email. Please do not reply.
"""

    return {
        "subject": "Your TradeMind password reset OTP",
        "html": html,
        "text": text,
    }
