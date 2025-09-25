from __future__ import annotations

import asyncio as aio
import pathlib
import typing as t

import httpx
import tenacity

ContentLike: t.TypeAlias = t.Union[str, bytes, t.Iterable[bytes], t.AsyncIterable[bytes]]


class DownloadTask(t.NamedTuple):
    url: str | httpx.URL
    save_to: t.Tuple[pathlib.Path, ...]
    method: t.Literal["GET", "POST"] = "GET"
    content: ContentLike | None = None
    data: t.Dict[str, t.Any] | None = None
    headers: httpx.Headers | None = None
    params: httpx.QueryParams | None = None
    cookies: httpx.Cookies | None = None
    auth: httpx.Auth | None = None
    save_url: str | None = None
    limit_bytes: int | None = None
    timeout: float = 10.0
    max_retries: int = 3
    retry_wait: float = 1.0


class HTTPDownloader:

    def __init__(self,
                 base_url: str | None = None,
                 max_connections: int = 100,
                 max_keepalive: int = 20,
                 connect_timeout: float = 5.0,
                 read_timeout: float = 10.0,
                 write_timeout: float = 10.0,
                 pool_timeout: float = 5.0,
                 http2: bool = False,
                 proxy: str | None = None,
                 concurrency: int = 10):
        limits = httpx.Limits(max_connections=max_connections,
                              max_keepalive_connections=max_keepalive)
        timeout = httpx.Timeout(connect=connect_timeout,
                                read=read_timeout,
                                write=write_timeout,
                                pool=pool_timeout)
        self.http_client = httpx.AsyncClient(base_url=base_url or "",
                                             limits=limits,
                                             timeout=timeout,
                                             http1=not http2,
                                             http2=http2,
                                             proxy=proxy)
        self.semaphore = aio.Semaphore(concurrency)

    async def download(self,
                       url: str | httpx.URL,
                       *save_to: pathlib.Path,
                       method: t.Literal["GET", "POST"] = "GET",
                       content: ContentLike | None = None,
                       data: t.Dict[str, t.Any] | None = None,
                       headers: httpx.Headers | None = None,
                       params: httpx.QueryParams | None = None,
                       cookies: httpx.Cookies | None = None,
                       auth: httpx.Auth | None = None,
                       save_url: str | None = None,
                       limit_bytes: int | None = None,
                       timeout: float = 10.0,
                       max_retries: int = 3,
                       retry_wait: float = 1.0) -> bytes:

        @tenacity.retry(stop=tenacity.stop_after_attempt(max_retries),
                        wait=tenacity.wait_fixed(retry_wait),
                        reraise=True)
        async def _download() -> bytes:
            async with self.http_client.stream(method,
                                               url,
                                               content=content,
                                               data=data,
                                               headers=headers,
                                               params=params,
                                               cookies=cookies,
                                               auth=auth,
                                               timeout=timeout) as response:
                response = t.cast(httpx.Response, response)
                if response.status_code != httpx.codes.OK:
                    raise httpx.HTTPStatusError(
                        f"Failed to download {url}: "
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response)

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    if limit_bytes and len(buffer) + len(chunk) > limit_bytes:
                        raise ValueError("Downloaded content exceeds the specified limit.")
                    buffer.extend(chunk)

                if not buffer:
                    raise ValueError("Downloaded content is empty.")
                return bytes(buffer)

        async with self.semaphore:
            data = await _download()
            if save_url:
                for dest in save_to:
                    file_dest = dest / save_url
                    file_dest.parent.mkdir(parents=True, exist_ok=True)
                    file_dest.write_bytes(data)
            return data

    async def downloads(self, *tasks: DownloadTask) -> t.List[bytes]:
        return await aio.gather(*[
            self.download(t_.url,
                          *t_.save_to,
                          method=t_.method,
                          content=t_.content,
                          data=t_.data,
                          headers=t_.headers,
                          params=t_.params,
                          cookies=t_.cookies,
                          auth=t_.auth,
                          save_url=t_.save_url,
                          limit_bytes=t_.limit_bytes,
                          timeout=t_.timeout,
                          max_retries=t_.max_retries,
                          retry_wait=t_.retry_wait) for t_ in tasks
        ])
