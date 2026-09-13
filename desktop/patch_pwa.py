"""
PWA and iOS Native Web App Enabler for Streamlit.
Injects Apple Mobile Web App tags, apple-touch-icon, and manifest into Streamlit's index.html
so that tapping "Add to Home Screen" on iPhone creates an authentic native iOS app.
"""
import os
import sys
import shutil
import logging

logger = logging.getLogger("PWAEnabler")


def patch_streamlit_pwa():
    try:
        import streamlit
        st_dir = os.path.dirname(streamlit.__file__)
        static_dir = os.path.join(st_dir, "static")
        index_html = os.path.join(static_dir, "index.html")

        if not os.path.exists(index_html):
            logger.warning("Streamlit static index.html not found.")
            return False

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_static = os.path.join(project_root, "static")

        # Copy icon and manifest to Streamlit static folder
        for fname in ["apple-touch-icon.png", "manifest.json", "icon-192.png", "icon-512.png"]:
            src = os.path.join(source_static, fname)
            dst = os.path.join(static_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Also copy apple-touch-icon as favicon.png
        icon_src = os.path.join(source_static, "apple-touch-icon.png")
        if os.path.exists(icon_src):
            shutil.copy2(icon_src, os.path.join(static_dir, "favicon.png"))

        # Read and check index.html
        with open(index_html, "r", encoding="utf-8") as f:
            html = f.read()

        if "apple-mobile-web-app-capable" in html:
            logger.info("Streamlit index.html already has iOS PWA tags.")
            return True

        pwa_tags = """
    <!-- iOS & Android Native PWA Standalone Mode -->
    <meta name="apple-mobile-web-app-capable" content="yes" />
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
    <meta name="apple-mobile-web-app-title" content="NSE Stocks" />
    <meta name="mobile-web-app-capable" content="yes" />
    <meta name="theme-color" content="#080b11" />
    <link rel="apple-touch-icon" href="./apple-touch-icon.png" />
    <link rel="manifest" href="./manifest.json" />
"""
        # Inject right after <head>
        head_pos = html.find("<head>")
        if head_pos != -1:
            new_html = html[: head_pos + 6] + pwa_tags + html[head_pos + 6 :]
            with open(index_html, "w", encoding="utf-8") as f:
                f.write(new_html)
            logger.info("Successfully injected iOS PWA tags into Streamlit index.html!")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to patch Streamlit PWA: {e}")
        return False


if __name__ == "__main__":
    success = patch_streamlit_pwa()
    print("PWA Patch Success:", success)
