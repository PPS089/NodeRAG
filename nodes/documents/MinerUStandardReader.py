import os
import time
import json
import hashlib
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv
from utils.FindProjectRoot import find_project_root as fr

class MinerUStandardReader:
    """
    MinerU 标准精准解析 API，本地文件上传版：

    - 输入是本地文件路径
    - 本地只负责读取文件并上传到 MinerU 提供的签名 URL
    - 不在本地做 MinerU 解析
    - 解析任务仍然由 MinerU API 完成
    - 支持缓存，避免同一个文件重复解析
    - 支持用户确认后重新解析
    """

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 600,
        interval: int = 5,
        cache_dir: Optional[str] = None,  # 缓存目录
    ):
        # 获取项目文件根目录
        project_root = fr()
        # 设置 .env 文件
        load_dotenv(project_root / ".env")

        # 缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else project_root / ".mineru_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 优先使用外部传入的 token
        # 如果没有传入，则读取环境变量 MINERU_API_TOKEN
        self.token = token or os.getenv("MINERU_API_TOKEN")

        # 如果没有 token，就直接报错
        if not self.token:
            raise ValueError("请设置 MINERU_API_TOKEN 环境变量")

        # 保存最大等待解析完成的时间
        self.timeout = timeout

        # 保存每次轮询任务状态的间隔
        self.interval = interval

        # 构造 MinerU API 请求头
        self.headers = {
            # 请求体格式是 JSON
            "Content-Type": "application/json",

            # Bearer Token 鉴权
            "Authorization": f"Bearer {self.token}",
        }


    def read_file(
        self,
        file_path: str,
        model_version: str = "vlm",
        language: str = "ch",
        is_ocr: bool = False,
        enable_table: bool = True,
        enable_formula: bool = True,
        page_ranges: Optional[str] = None,
        force_reparse: bool = False,
        confirm_reparse: bool = False,
    ) -> str:
        """
        读取本地文件，并通过 MinerU API 解析。

        缓存逻辑：

        1. 先计算本地文件内容 hash
        2. 再把文件 hash + 解析参数组成 cache_key
        3. 如果 cache_key 已存在，默认直接返回缓存
        4. 如果 confirm_reparse=True，命中缓存时询问用户是否重新解析
        5. 如果 force_reparse=True，直接重新上传并重新解析
        """

        # 把传入的文件路径转换成 Path 对象
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查路径是否真的是文件，而不是目录
        if not path.is_file():
            raise ValueError(f"不是有效文件: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持本地 PDF 文件: {file_path}")

        # 计算本地文件内容 hash
        # 这样即使文件名一样，只要内容不同，也会认为是不同文件
        file_hash = self._hash_file(path)

        # 构造提交给 MinerU 的解析参数
        payload = {
            # 本地文件名，MinerU 用它识别文件名和后缀
            "file_name": path.name,

            # MinerU 模型版本
            "model_version": model_version,

            # 文档语言
            "language": language,

            # 是否强制 OCR
            "is_ocr": is_ocr,

            # 是否启用表格识别
            "enable_table": enable_table,

            # 是否启用公式识别
            "enable_formula": enable_formula,
        }

        # 如果用户指定了解析页码范围，则加入请求参数
        if page_ranges:
            payload["page_ranges"] = page_ranges

        # 构造用于生成缓存 key 的信息
        # 注意这里额外加入了 file_hash
        # 因为本地文件没有稳定 URL，所以必须用文件内容 hash 判断是否是同一个文件
        cache_identity = {
            "file_hash": file_hash,
            "payload": payload,
        }

        # 根据文件 hash 和解析参数生成缓存 key
        cache_key = self._make_cache_key(cache_identity)

        # Markdown 缓存路径
        cache_path = self.cache_dir / f"{cache_key}.md"

        # 元数据缓存路径
        meta_path = self.cache_dir / f"{cache_key}.json"

        # 如果缓存存在，并且用户没有强制重新解析
        if cache_path.exists() and not force_reparse:
            # 如果需要用户确认是否重新解析
            if confirm_reparse:
                user_input = input(
                    "该本地文件使用相同参数已经解析过，是否重新解析？输入 y 重新解析，其他输入使用缓存："
                ).strip().lower()

                # 用户没有输入 y，则直接返回缓存
                if user_input != "y":
                    return cache_path.read_text(encoding="utf-8")

                # 用户输入 y，则继续往下走，重新调用 MinerU API

            else:
                # 默认不重复解析，直接返回缓存
                return cache_path.read_text(encoding="utf-8")

        # 第一步：创建文件解析任务，并获取 MinerU 返回的签名上传 URL
        upload_url, task_id = self._create_file_task(payload)

        # 第二步：把本地文件上传到 MinerU 提供的签名 URL
        self._upload_file(upload_url, path)

        # 第三步：根据 task_id 轮询解析任务状态，直到拿到 zip 下载地址
        zip_url = self._poll_zip_url(task_id)

        # 第四步：下载 zip 结果包，并读取 full.md
        markdown = self._download_full_md(zip_url, cache_key)

        # 把 Markdown 结果写入缓存
        cache_path.write_text(markdown, encoding="utf-8")

        # 保存缓存元数据，方便排查
        meta = {
            # MinerU 任务 ID
            "task_id": task_id,

            # 本地文件路径
            "file_path": str(path),

            # 文件名
            "file_name": path.name,

            # 文件内容 hash
            "file_hash": file_hash,

            # 解析参数
            "payload": payload,

            # 缓存 key
            "cache_key": cache_key,

            # 缓存创建时间
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 写入 JSON 元数据文件
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 返回解析后的 Markdown
        return markdown

    def _create_file_task(self, payload: dict) -> tuple[str, str]:
        """
        创建 MinerU 本地文件解析任务。

        返回：
        - upload_url：MinerU 提供的签名上传 URL
        - task_id：解析任务 ID
        """

        # 调用 MinerU 文件解析任务创建接口
        resp = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            headers=self.headers,
            json={
                "enable_formula": payload["enable_formula"],
                "enable_table": payload["enable_table"],
                "language": payload["language"],
                "model_version": payload["model_version"],
                "files": [
                    {
                        "name": payload["file_name"],
                        "is_ocr": payload["is_ocr"],
                        **({"page_ranges": payload["page_ranges"]} if "page_ranges" in payload else {}),
                    }
                ],
            },
            timeout=30,
        )

        # 检查响应
        result = self._check_response(resp)

        # 取出 data 字段
        data = result["data"]

        # 从响应里取出任务 ID
        task_id = data["batch_id"]

        # 从响应里取出文件上传 URL
        # 不同 MinerU API 返回字段可能叫 file_url 或 upload_url
        upload_url = data["file_urls"][0]

        # 如果没有拿到上传 URL，说明接口返回结构不符合预期
        if not upload_url:
            raise RuntimeError(f"MinerU 未返回文件上传 URL: {result}")

        # 返回上传 URL 和任务 ID
        return upload_url, task_id

    @staticmethod
    def _upload_file(upload_url: str, path: Path) -> None:
        """
        上传本地文件到 MinerU 提供的签名 URL。
        """

        # 以二进制方式打开本地文件
        with path.open("rb") as f:
            # 使用 PUT 请求上传文件内容
            resp = requests.put(
                upload_url,
                data=f,
                timeout=300,
            )

        # 如果上传状态码不是 200 或 201，认为上传失败
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"文件上传失败: HTTP {resp.status_code}, {resp.text[:500]}"
            )

    def _poll_zip_url(self, task_id: str) -> str:
        """
        轮询 MinerU 任务状态，直到解析完成并返回 zip 下载地址。
        """

        # 记录开始时间
        start = time.time()

        # 在超时时间内持续轮询
        while time.time() - start < self.timeout:
            # 查询任务状态
            resp = requests.get(
                f"https://mineru.net/api/v4/extract-results/batch/{task_id}",
                headers=self.headers,
                timeout=30,
            )

            # 检查响应
            result = self._check_response(resp)

            # 获取 data 字段
            data = result["data"]

            # 获取解析结果列表
            extract_results = data.get("extract_result", [])

            # 如果还没有解析结果，继续等待
            if not extract_results:
                time.sleep(self.interval)
                continue

            # 当前只上传了一个本地 PDF，所以取第一个结果
            first_result = extract_results[0]

            # 获取任务状态
            state = first_result.get("state")

            # 如果任务完成
            if state == "done":
                # 返回完整结果 zip 包地址
                return first_result["full_zip_url"]

            # 如果任务失败
            if state == "failed":
                raise RuntimeError(f"MinerU 解析失败: {first_result.get('err_msg', '未知错误')}")

            # 如果任务还在处理中，等待一段时间后继续查询
            time.sleep(self.interval)

        # 超过最大等待时间则抛出异常
        raise TimeoutError(f"MinerU 解析超时，task_id={task_id}")

    def _download_full_md(self, zip_url: str, cache_key: str) -> str:
        """
        下载 MinerU 返回的 zip 文件，并读取其中的 full.md。
        """

        # 下载 zip 文件
        resp = requests.get(zip_url, timeout=120)

        # 如果 HTTP 状态码异常，抛出错误
        resp.raise_for_status()

        # 每个 PDF 的完整解析结果保存目录
        extract_dir = self.cache_dir / cache_key

        # 创建完整解析结果保存目录
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 保存 MinerU 返回的原始 zip 文件
        zip_path = extract_dir / "mineru_result.zip"

        # 写入原始 zip 文件
        zip_path.write_bytes(resp.content)

        # 使用 zipfile 打开下载到的 zip 二进制内容
        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            # 解压整个 zip 文件，里面会包含图片、JSON、Markdown 等结果
            z.extractall(extract_dir)

            # 查找 zip 包中以 full.md 结尾的文件
            md_names = [
                name
                for name in z.namelist()
                if name.endswith("full.md")
            ]

            # 如果没有找到 full.md，则报错
            if not md_names:
                raise RuntimeError(f"zip 中未找到 full.md，文件列表: {z.namelist()}")

            # 读取 full.md，并按 utf-8 解码成字符串
            return z.read(md_names[0]).decode("utf-8")

    @staticmethod
    def _check_response(resp: requests.Response) -> dict:
        """
        检查 HTTP 响应和 MinerU 业务响应是否成功。
        """

        # 检查 HTTP 状态码
        resp.raise_for_status()

        # 解析 JSON 响应
        result = resp.json()

        # MinerU 业务成功一般是 code == 0
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU API 返回错误: {result}")

        # 返回响应字典
        return result

    @staticmethod
    def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        """
        计算本地文件 SHA256。

        使用分块读取，避免大文件一次性读入内存。
        """

        # 创建 sha256 哈希对象
        sha256 = hashlib.sha256()

        # 以二进制方式打开文件
        with path.open("rb") as f:
            # 循环读取文件块
            while True:
                # 每次读取 chunk_size 字节
                chunk = f.read(chunk_size)

                # 如果读不到内容，说明文件读完了
                if not chunk:
                    break

                # 把当前文件块加入 hash 计算
                sha256.update(chunk)

        # 返回十六进制 hash 字符串
        return sha256.hexdigest()

    @staticmethod
    def _make_cache_key(data: dict) -> str:
        """
        根据文件 hash 和解析参数生成缓存 key。
        """

        # 把 data 转成稳定 JSON 字符串
        data_str = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
        )

        # 用 sha256 生成缓存 key
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()