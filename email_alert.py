"""
email_alert.py
--------------
Sends an SMTP email alert when a HIGH-severity pothole is detected.

Configuration:
  Set environment variables (recommended) or edit the defaults below:
    POTHOLE_SENDER_EMAIL    - Gmail address used to send alerts
    POTHOLE_SENDER_PASSWORD - App password (NOT your Gmail login password)
    POTHOLE_RECEIVER_EMAIL  - Maintenance authority email address
    POTHOLE_SMTP_HOST       - (optional) default: smtp.gmail.com
    POTHOLE_SMTP_PORT       - (optional) default: 587

Gmail app-password guide:
  https://support.google.com/accounts/answer/185833

Fixes applied:
  - Changed MIMEMultipart("alternative") -> MIMEMultipart("mixed") so binary
    image attachments are correctly sent (alternative only allows text parts).
  - Added is_configured() helper so callers can warn before attempting SMTP.
  - Placeholder credential check: logs a clear warning instead of attempting
    a doomed SMTP connection when env vars are not set.
"""

import os
import smtplib
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Load .env file for persistent credentials (auto-loads on import)
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
# Email configuration  (use environment vars for security)
# ──────────────────────────────────────────────
SENDER_EMAIL    = os.environ.get("POTHOLE_SENDER_EMAIL",    "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("POTHOLE_SENDER_PASSWORD", "your_app_password")
RECEIVER_EMAIL  = os.environ.get("POTHOLE_RECEIVER_EMAIL",  "maintenance@city.gov")
SMTP_HOST       = os.environ.get("POTHOLE_SMTP_HOST",       "smtp.gmail.com")
SMTP_PORT       = int(os.environ.get("POTHOLE_SMTP_PORT",   "587"))

# Placeholder values that indicate email is NOT configured
_PLACEHOLDERS = {"your_email@gmail.com", "your_app_password"}


def is_configured() -> bool:
    """Return True only if real (non-placeholder) credentials are set."""
    return (
        SENDER_EMAIL    not in _PLACEHOLDERS and
        SENDER_PASSWORD not in _PLACEHOLDERS and
        bool(SENDER_EMAIL) and
        bool(SENDER_PASSWORD)
    )


def send_alert(
    severity: str,
    confidence: float,
    pothole_count: int,
    image_path: str | None = None,
    location: str = "Unknown Road Segment",
) -> bool:
    """
    Send a maintenance alert email.

    Parameters
    ----------
    severity      : Classified severity - expected "High"
    confidence    : Detection confidence score (0-1)
    pothole_count : Number of potholes detected in the image
    image_path    : Optional path to the annotated result image (attached)
    location      : Human-readable location string

    Returns
    -------
    bool - True if email sent successfully, False otherwise
    """
    if str(severity).strip().lower() != "high":
        # Only trigger alert for HIGH severity
        return False

    # FIX: Early-exit with clear message if credentials are placeholders
    if not is_configured():
        print("[EMAIL] Alert skipped - email not configured.")
        print("        Set POTHOLE_SENDER_EMAIL, POTHOLE_SENDER_PASSWORD,")
        print("        POTHOLE_RECEIVER_EMAIL env vars and restart app.py.")
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject   = "URGENT: High Severity Pothole Detected"

    # ── Build HTML body ──────────────────────────────────────────
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background:#f4f4f4; padding:20px;">
        <div style="max-width:600px; margin:auto; background:#fff;
                    border-radius:8px; overflow:hidden;
                    box-shadow:0 2px 8px rgba(0,0,0,0.15);">

          <!-- Header -->
          <div style="background:#c0392b; padding:20px; text-align:center;">
            <h1 style="color:#fff; margin:0; font-size:22px;">
              HIGH SEVERITY POTHOLE DETECTED
            </h1>
            <p style="color:#f5b7b1; margin:4px 0 0;">
              Immediate maintenance action required
            </p>
          </div>

          <!-- Body -->
          <div style="padding:24px;">
            <table style="width:100%; border-collapse:collapse;">
              <tr>
                <td style="padding:10px; font-weight:bold; color:#555;
                            border-bottom:1px solid #eee; width:40%;">
                  Severity Level
                </td>
                <td style="padding:10px; color:#c0392b; font-weight:bold;
                            border-bottom:1px solid #eee;">
                  {str(severity).upper()}
                </td>
              </tr>
              <tr>
                <td style="padding:10px; font-weight:bold; color:#555;
                            border-bottom:1px solid #eee;">
                  Detection Confidence
                </td>
                <td style="padding:10px; border-bottom:1px solid #eee;">
                  {confidence * 100:.1f}%
                </td>
              </tr>
              <tr>
                <td style="padding:10px; font-weight:bold; color:#555;
                            border-bottom:1px solid #eee;">
                  Potholes Detected
                </td>
                <td style="padding:10px; border-bottom:1px solid #eee;">
                  {pothole_count}
                </td>
              </tr>
              <tr>
                <td style="padding:10px; font-weight:bold; color:#555;
                            border-bottom:1px solid #eee;">
                  Location
                </td>
                <td style="padding:10px; border-bottom:1px solid #eee;">
                  {location}
                </td>
              </tr>
              <tr>
                <td style="padding:10px; font-weight:bold; color:#555;">
                  Timestamp
                </td>
                <td style="padding:10px;">
                  {timestamp}
                </td>
              </tr>
            </table>

            <!-- Recommendation box -->
            <div style="margin-top:20px; background:#fef9e7;
                        border-left:4px solid #f39c12;
                        padding:14px 18px; border-radius:4px;">
              <strong style="color:#d35400;">Maintenance Recommendation</strong>
              <ul style="margin:8px 0 0; padding-left:20px; color:#555;">
                <li>Dispatch road repair crew within 24 hours</li>
                <li>Place warning signs / cones at the location</li>
                <li>Conduct full road survey in the area</li>
                <li>Prioritise permanent patching over temporary fills</li>
              </ul>
            </div>
          </div>

          <!-- Footer -->
          <div style="background:#ecf0f1; padding:14px;
                      text-align:center; font-size:12px; color:#888;">
            This alert was generated automatically by the
            <strong>Smart Pothole Detection System</strong>.<br>
            Please do not reply to this email.
          </div>
        </div>
      </body>
    </html>
    """

    # ── Assemble MIME message ────────────────────────────────────
    # FIX: Use "mixed" (not "alternative") when attaching binary files.
    # "alternative" is for text/html vs text/plain variants only.
    msg = MIMEMultipart("mixed")
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECEIVER_EMAIL
    msg["Subject"] = subject

    # HTML body wrapped in a "related" sub-part
    body_part = MIMEMultipart("alternative")
    body_part.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(body_part)

    # Attach result image if provided
    if image_path and os.path.isfile(image_path):
        try:
            with open(image_path, "rb") as f:
                part = MIMEBase("image", "jpeg")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(image_path)}"',
            )
            msg.attach(part)
            print(f"[EMAIL] Attaching image: {os.path.basename(image_path)}")
        except Exception as attach_err:
            print(f"[EMAIL WARNING] Could not attach image: {attach_err}")
            # Non-critical – continue sending without attachment

    # ── Send via SMTP ────────────────────────────────────────────
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"[EMAIL] Alert sent successfully to {RECEIVER_EMAIL} at {timestamp}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[EMAIL ERROR] Authentication failed.")
        print("              Ensure POTHOLE_SENDER_PASSWORD is a Gmail App Password,")
        print("              not your regular Gmail password.")
        print("              Guide: https://myaccount.google.com/apppasswords")
        return False

    except smtplib.SMTPException as smtp_err:
        print(f"[EMAIL ERROR] SMTP error: {smtp_err}")
        return False

    except Exception as exc:
        print(f"[EMAIL ERROR] Unexpected error: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────
# Quick self-test (python email_alert.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Email Configuration Check")
    print(f"  Sender   : {SENDER_EMAIL}")
    print(f"  Receiver : {RECEIVER_EMAIL}")
    print(f"  SMTP     : {SMTP_HOST}:{SMTP_PORT}")
    print(f"  Configured: {is_configured()}")

    if not is_configured():
        print("\n[SKIP] Email credentials not configured.")
        print("       Set POTHOLE_SENDER_EMAIL, POTHOLE_SENDER_PASSWORD,")
        print("       POTHOLE_RECEIVER_EMAIL env vars to enable email alerts.")
    else:
        print("\n[TEST] Sending test HIGH-severity alert...")
        ok = send_alert(
            severity="High",
            confidence=0.94,
            pothole_count=3,
            image_path=None,
            location="Test Road, City Centre",
        )
        print("[TEST] Result:", "Sent successfully" if ok else "Failed - check logs above")
