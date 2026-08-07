"""
security/email.py — OTP Email Sender
======================================
Sends secure 6-digit OTP codes via SMTP for Multi-Factor Authentication.
Uses Python's built-in smtplib — no external packages required.
"""

import logging
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import get_settings

logger = logging.getLogger(__name__)


def generate_otp(digits: int = 6) -> str:
    """Generates a cryptographically secure numeric OTP."""
    return "".join([str(secrets.randbelow(10)) for _ in range(digits)])


def send_otp_email(to_address: str, username: str, otp_code: str) -> bool:
    """
    Sends a formatted OTP email to the user's registered email address.
    Returns True if sent successfully, False otherwise.
    """
    settings = get_settings()

    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "SMTP credentials not configured. OTP code for %s: %s",
            to_address,
            otp_code,
        )
        # In dev mode without SMTP, log the OTP and return True so the flow works
        return True

    subject = "Your Axion Login Code"
    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;font-family:Inter,Arial,sans-serif;background:#0d1117;">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td align="center" style="padding:40px 20px;">
        <table width="520" cellpadding="0" cellspacing="0" style="
            background:#161b22;
            border:1px solid #21262d;
            border-radius:16px;
            overflow:hidden;
          ">
          <!-- Header -->
          <tr>
            <td style="
                background:linear-gradient(135deg,#3b82f6,#1d4ed8);
                padding:28px 32px;
              ">
              <h1 style="margin:0;color:#fff;font-size:22px;font-weight:800;letter-spacing:-0.5px;">
                🔐 Axion Security Code
              </h1>
              <p style="margin:6px 0 0;color:rgba(255,255,255,0.8);font-size:14px;">
                Your one-time login verification code
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:32px;">
              <p style="margin:0 0 24px;color:#c9d1d9;font-size:15px;line-height:1.6;">
                Hi <strong style="color:#fff;">{username}</strong>,
                <br>Use the code below to complete your login. This code expires in
                <strong style="color:#3b82f6;">5 minutes</strong>.
              </p>

              <!-- OTP Box -->
              <div style="
                  background:#0d1117;
                  border:2px solid #3b82f6;
                  border-radius:12px;
                  text-align:center;
                  padding:24px;
                  margin:0 0 24px;
                ">
                <span style="
                    font-size:42px;
                    font-weight:900;
                    letter-spacing:12px;
                    color:#3b82f6;
                    font-family:monospace;
                  ">{otp_code}</span>
              </div>

              <p style="margin:0;color:#8b949e;font-size:13px;line-height:1.6;">
                ⚠️ Never share this code. If you didn't request this, ignore this email and your account remains secure.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="
                border-top:1px solid #21262d;
                padding:20px 32px;
              ">
              <p style="margin:0;color:#484f58;font-size:12px;text-align:center;">
                Axion AI Investment Advisor &mdash; Sent automatically, do not reply.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM
    msg["To"] = to_address
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, [to_address], msg.as_string())
        logger.info("OTP email sent to %s", to_address)
        return True
    except Exception as exc:
        logger.error("Failed to send OTP email to %s: %s", to_address, exc)
        return False
