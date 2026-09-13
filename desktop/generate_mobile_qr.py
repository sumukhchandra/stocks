"""
Generates a styled QR code for connecting mobile phones to the NSE Stocks AI terminal.
"""
import os
import socket
import qrcode
import logging

logger = logging.getLogger("MobileQR")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "desktop", "mobile_qr.png")


def get_local_ip() -> str:
    """Detect local WiFi / LAN IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def generate_qr(port: int = 8501) -> str:
    """Generates QR code PNG and returns the mobile access URL."""
    ip = get_local_ip()
    mobile_url = f"http://{ip}:{port}"

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(mobile_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#00e5ff", back_color="#0d121c")
    img.save(OUTPUT_PATH)
    logger.info(f"Mobile QR code saved to {OUTPUT_PATH} for {mobile_url}")
    return mobile_url


if __name__ == "__main__":
    url = generate_qr()
    print(f"Mobile URL: {url}")
