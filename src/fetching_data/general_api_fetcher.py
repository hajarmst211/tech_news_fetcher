import logging
import requests
from typing import Optional, Dict, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralApiFetcher:
    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        ssl_verify: bool = True
    ):
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.timeout = timeout

        self.session = requests.Session()
        self.session.verify = ssl_verify

        default_headers = {
            "User-Agent": "TechNewsPipeline/1.0",
            "Accept": "application/json"
        }
        self.session.headers.update(default_headers)

        if headers:
            self.session.headers.update(headers)

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

    def _build_url(self, endpoint: str) -> str:
        url = f"{self.base_url}{endpoint}" if endpoint.startswith("/") else endpoint
        if not url.startswith("http"):
            url = f"{self.base_url}/{endpoint}" if self.base_url else endpoint
        return url

    def _send(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[requests.Response]:
        url = self._build_url(endpoint)
        try:
            logger.info(f"Sending {method} request to: {url}")
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_data,
                headers=extra_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info(f"  [OK] {response.status_code} from {method} {url}")
            return response
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error {response.status_code} on {method} {url}: {http_err}")
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred ({self.timeout}s) on {method} {url}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Network or request error on {method} {url}: {req_err}")
        return None

    def request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Union[Dict[str, Any], list]]:
        response = self._send(endpoint, method, params, json_data, extra_headers)
        if response is None:
            return None
        try:
            return response.json()
        except ValueError as json_err:
            logger.error(f"Failed to decode JSON from {response.url}: {json_err}")
            return None

    def request_raw(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        response = self._send(endpoint, method, params, json_data, extra_headers)
        if response is None:
            return None
        return response.text

    def request_response(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[requests.Response]:
        return self._send(endpoint, method, params, json_data, extra_headers)

    def close(self):
        self.session.close()