"""
HTML Render HTTP Service

Sandboxed HTML rendering service using Playwright for capturing screenshots.
Used by the RL reward function to validate generated HTML code.

Features:
- FastAPI-based HTTP service
- Playwright Chromium for headless rendering
- Auto-shutdown after idle timeout
- Batch rendering support

Usage:
    python html_render_service.py --host 0.0.0.0 --port 8768

Or with uvicorn:
    uvicorn html_render_service:app --host 0.0.0.0 --port 8768 --workers 4
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import asyncio
import os
import time
import traceback
from pathlib import Path
from playwright.async_api import async_playwright
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HTML Render Service", version="1.0.0")

# Global Playwright browser instance (reused for performance)
browser_instance = None
browser_lock = asyncio.Lock()

# Auto-shutdown configuration
_last_request_time = None
_idle_timeout_seconds = int(os.environ.get("RENDER_SERVICE_IDLE_TIMEOUT", 600))
_auto_shutdown_enabled = os.environ.get("RENDER_SERVICE_AUTO_SHUTDOWN", "true").lower() == "true"
_idle_checker_task = None


class RenderRequest(BaseModel):
    """Render request model."""
    html_filepath: str  # Absolute path to HTML file
    screenshot_filepath: str  # Path to save screenshot
    width: int = 800
    height: int = 600
    timeout: int = 25000  # Timeout in milliseconds


class RenderResponse(BaseModel):
    """Render response model."""
    success: bool
    screenshot_filepath: Optional[str] = None
    error: Optional[str] = None
    render_time: float = 0.0


async def get_browser():
    """Get global browser instance (singleton pattern)."""
    global browser_instance

    async with browser_lock:
        if browser_instance is None:
            logger.info("Initializing Playwright browser...")
            playwright = await async_playwright().start()
            browser_instance = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-gpu'
                ]
            )
            logger.info("Browser initialized successfully")

    return browser_instance


async def _idle_checker():
    """Periodically check for idle timeout and auto-shutdown."""
    global _last_request_time

    while True:
        await asyncio.sleep(30)

        if not _auto_shutdown_enabled:
            continue

        if _last_request_time is not None:
            idle_time = time.time() - _last_request_time
            if idle_time > _idle_timeout_seconds:
                logger.info(f"Service idle for {_idle_timeout_seconds}s, shutting down...")
                await asyncio.sleep(1)
                os._exit(0)


@app.on_event("startup")
async def startup_event():
    """Initialize browser on service startup."""
    global _last_request_time, _idle_checker_task

    logger.info("HTML Render Service starting...")
    logger.info(f"  Auto-shutdown: {'enabled' if _auto_shutdown_enabled else 'disabled'}")
    logger.info(f"  Idle timeout: {_idle_timeout_seconds}s")

    await get_browser()
    _last_request_time = time.time()

    if _auto_shutdown_enabled:
        _idle_checker_task = asyncio.create_task(_idle_checker())

    logger.info("Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup browser on shutdown."""
    global browser_instance
    if browser_instance:
        logger.info("Closing browser...")
        await browser_instance.close()
        browser_instance = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "HTML Render Service",
        "status": "running",
        "browser_ready": browser_instance is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "browser_initialized": browser_instance is not None,
        "auto_shutdown_enabled": _auto_shutdown_enabled,
        "idle_timeout_seconds": _idle_timeout_seconds,
        "last_request_time": _last_request_time,
        "idle_seconds": time.time() - _last_request_time if _last_request_time else None
    }


@app.post("/shutdown")
async def shutdown():
    """Manual shutdown endpoint."""
    logger.info("Manual shutdown requested...")

    async def delayed_shutdown():
        await asyncio.sleep(1)
        os._exit(0)

    asyncio.create_task(delayed_shutdown())
    return {"status": "shutting_down"}


@app.post("/render", response_model=RenderResponse)
async def render_html(request: RenderRequest):
    """
    Render HTML file to screenshot.

    Args:
        request: RenderRequest with file paths and dimensions

    Returns:
        RenderResponse with success status and render time
    """
    global _last_request_time
    _last_request_time = time.time()

    start_time = time.time()
    context = None
    page = None

    async def _do_render():
        nonlocal context, page

        # Validate HTML file exists
        html_path = Path(request.html_filepath)
        if not html_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"HTML file not found: {request.html_filepath}"
            )

        logger.info(f"Rendering: {request.html_filepath}")

        browser = await get_browser()

        # Create browser context and page
        context = await browser.new_context(
            viewport={"width": request.width, "height": request.height}
        )

        page = await context.new_page()
        page.set_default_timeout(request.timeout)

        # Load HTML file
        file_url = f"file://{os.path.abspath(request.html_filepath)}"
        await page.goto(file_url, timeout=request.timeout, wait_until="domcontentloaded")

        # Wait for network idle (with timeout protection)
        try:
            await page.wait_for_load_state("networkidle", timeout=request.timeout)
        except Exception as e:
            logger.warning(f"Network idle timeout, using domcontentloaded: {e}")

        # Ensure screenshot directory exists
        screenshot_path = Path(request.screenshot_filepath)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Take screenshot
        await page.screenshot(
            path=request.screenshot_filepath,
            type='png',
            full_page=True,
            timeout=request.timeout
        )

        render_time = time.time() - start_time
        logger.info(f"Rendered: {request.screenshot_filepath} ({render_time:.2f}s)")

        # Cleanup
        try:
            if page and not page.is_closed():
                await page.close()
            if context:
                await context.close()
        except Exception:
            pass

        return RenderResponse(
            success=True,
            screenshot_filepath=request.screenshot_filepath,
            render_time=render_time
        )

    try:
        timeout_seconds = (request.timeout / 1000.0) + 5.0
        return await asyncio.wait_for(_do_render(), timeout=timeout_seconds)

    except asyncio.TimeoutError:
        # Cleanup on timeout
        if page:
            try:
                await asyncio.wait_for(page.close(), timeout=1.0)
            except:
                pass
        if context:
            try:
                await asyncio.wait_for(context.close(), timeout=1.0)
            except:
                pass

        return RenderResponse(
            success=False,
            error=f"Render timeout ({timeout_seconds:.1f}s)",
            render_time=time.time() - start_time
        )

    except HTTPException:
        raise

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)

        # Cleanup on error
        if page:
            try:
                await page.close()
            except:
                pass
        if context:
            try:
                await context.close()
            except:
                pass

        return RenderResponse(
            success=False,
            error=str(e),
            render_time=time.time() - start_time
        )


@app.post("/render_batch")
async def render_html_batch(requests: list[RenderRequest]):
    """
    Batch render multiple HTML files.

    Args:
        requests: List of RenderRequest objects

    Returns:
        List of RenderResponse objects
    """
    logger.info(f"Batch rendering {len(requests)} files")

    tasks = [render_html(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for result in results:
        if isinstance(result, Exception):
            responses.append(RenderResponse(
                success=False,
                error=str(result),
                render_time=0.0
            ))
        else:
            responses.append(result)

    success_count = sum(1 for r in responses if r.success)
    logger.info(f"Batch complete: {success_count}/{len(requests)} succeeded")

    return responses


def main():
    """Start the HTTP service."""
    import argparse

    parser = argparse.ArgumentParser(description="HTML Render HTTP Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8768, help="Listen port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting service on {args.host}:{args.port}")

    uvicorn.run(
        "html_render_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
