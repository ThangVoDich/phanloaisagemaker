import io
import json
import os
from typing import Any, Dict, List

import boto3
import streamlit as st
from PIL import Image, ImageDraw

DEFAULT_CONF_THRES = 0.25


def get_config() -> Dict[str, str]:
    region = st.secrets.get("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1"))
    endpoint_name = st.secrets.get("SAGEMAKER_ENDPOINT_NAME", os.getenv("SAGEMAKER_ENDPOINT_NAME", ""))

    aws_access_key_id = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
    aws_session_token = st.secrets.get("AWS_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN"))

    return {
        "region": region,
        "endpoint_name": endpoint_name,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
    }


@st.cache_resource
def get_runtime_client():
    cfg = get_config()

    if not cfg["endpoint_name"]:
        raise ValueError("Thiếu SAGEMAKER_ENDPOINT_NAME trong secrets hoặc environment variables.")

    client_kwargs = {
        "service_name": "sagemaker-runtime",
        "region_name": cfg["region"],
    }

    if cfg["aws_access_key_id"] and cfg["aws_secret_access_key"]:
        client_kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        client_kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
        if cfg["aws_session_token"]:
            client_kwargs["aws_session_token"] = cfg["aws_session_token"]

    return boto3.client(**client_kwargs)


def build_message(result: Dict[str, Any]) -> str:
    status = result.get("status", "UNKNOWN")
    missing = result.get("missing", [])
    shifted = result.get("shifted", [])
    size_abnormal = result.get("size_abnormal", [])

    if status == "OK":
        return "Sản phẩm đạt yêu cầu."

    parts = []
    if missing:
        parts.append("Thiếu part: " + ", ".join(missing))
    if shifted:
        parts.append("Lệch vị trí: " + ", ".join(shifted))
    if size_abnormal:
        parts.append("Kích thước box bất thường: " + ", ".join(size_abnormal))

    return " | ".join(parts) if parts else "Sản phẩm không đạt yêu cầu."


def draw_boxes(image: Image.Image, result: Dict[str, Any]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    shifted = set(result.get("shifted", []))
    size_abnormal = set(result.get("size_abnormal", []))

    for class_name, d in result.get("detections", {}).items():
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        conf_pct = d["conf"] * 100

        color = "green"
        if class_name in shifted or class_name in size_abnormal:
            color = "red"

        label = f"{class_name} {conf_pct:.1f}%"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

        label_w = max(170, 10 * len(label))
        label_h = 30

        label_x1 = x1
        label_y1 = max(0, y1 - label_h)
        label_x2 = x1 + label_w
        label_y2 = label_y1 + label_h

        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
        draw.text((label_x1 + 6, label_y1 + 6), label, fill="white")

    return img


def invoke_single_image(
    runtime,
    endpoint_name: str,
    uploaded_file,
    conf: float,
    show_reference: bool,
) -> Dict[str, Any]:
    image_bytes = uploaded_file.getvalue()
    image_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=image_bytes,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    result["filename"] = uploaded_file.name
    result["image_rgb"] = image_rgb
    result["image_annotated"] = draw_boxes(image_rgb, result)
    result["message"] = build_message(result)
    result["show_reference"] = show_reference
    result["conf"] = conf

    return result


def main() -> None:
    cfg = get_config()
    endpoint_name = cfg["endpoint_name"]
    region = cfg["region"]

    st.set_page_config(page_title="Product Inspection", layout="wide")
    st.title("kiểm tra vị trí sản phẩm bằng AWS SageMaker")
    st.caption(f"Suy luận thời gian thực qua SageMaker Endpoint: {endpoint_name or '[chưa cấu hình]'}")
    st.write(
        "Ứng dụng này upload ảnh lên giao diện Streamlit, "
        "sau đó gọi AWS SageMaker endpoint để suy luận và trả về kết quả kiểm tra OK/NG."
    )

    with st.expander("Thông tin hệ thống"):
        st.write(f"Region: {region}")
        st.write(f"Endpoint: {endpoint_name or 'Chưa cấu hình'}")

    try:
        runtime = get_runtime_client()
    except Exception as exc:
        st.error(f"Không khởi tạo được SageMaker runtime client: {exc}")
        st.info("Hãy cấu hình secrets trên Streamlit Cloud trước khi chạy app.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF_THRES, 0.05)
    show_reference = st.checkbox("Hiện vùng chuẩn", value=True)

    if not uploaded_files:
        st.info("Hãy upload ít nhất một ảnh để kiểm tra.")
        return

    results_all: List[Dict[str, Any]] = []

    with st.spinner("Đang gọi SageMaker endpoint cho nhiều ảnh..."):
        for uploaded_file in uploaded_files:
            try:
                result = invoke_single_image(
                    runtime=runtime,
                    endpoint_name=endpoint_name,
                    uploaded_file=uploaded_file,
                    conf=conf,
                    show_reference=show_reference,
                )
                results_all.append(result)
            except Exception as exc:
                uploaded_file.seek(0)
                fallback_image = Image.open(uploaded_file).convert("RGB")
                results_all.append(
                    {
                        "filename": uploaded_file.name,
                        "image_rgb": fallback_image,
                        "image_annotated": fallback_image,
                        "status": "ERROR",
                        "message": str(exc),
                        "detections": {},
                        "missing": [],
                        "shifted": [],
                        "size_abnormal": [],
                    }
                )

    st.subheader("Tổng hợp kết quả")
    ok_count = sum(1 for x in results_all if x["status"] == "OK")
    ng_count = sum(1 for x in results_all if x["status"] == "NG")
    err_count = sum(1 for x in results_all if x["status"] == "ERROR")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success(f"OK: {ok_count}")
    with col_b:
        st.error(f"NG: {ng_count}")
    with col_c:
        if err_count > 0:
            st.warning(f"ERROR: {err_count}")
        else:
            st.info("ERROR: 0")

    for item in results_all:
        st.markdown("---")
        st.subheader(item["filename"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                item["image_annotated"],
                caption=item["filename"],
                use_container_width=True,
            )

        with col2:
            if item["status"] == "OK":
                st.success("OK")
            elif item["status"] == "NG":
                st.error("NG")
            else:
                st.warning("ERROR")

            st.write("**Tóm tắt:**")
            st.write(item["message"])

            st.write("**Part detect được:**")
            if item.get("detections"):
                for class_name, d in item["detections"].items():
                    conf_pct = d["conf"] * 100
                    st.write(
                        f"- {class_name}: {conf_pct:.1f}% | "
                        f"x1={d['x1']:.1f}, y1={d['y1']:.1f}, "
                        f"x2={d['x2']:.1f}, y2={d['y2']:.1f}"
                    )
            else:
                st.write("- Không detect được part nào")

            if item.get("missing"):
                st.write("**Thiếu part:**")
                for x in item["missing"]:
                    st.write(f"- {x}")

            if item.get("shifted"):
                st.write("**Lệch vị trí:**")
                for x in item["shifted"]:
                    st.write(f"- {x}")

            if item.get("size_abnormal"):
                st.write("**Kích thước box bất thường:**")
                for x in item["size_abnormal"]:
                    st.write(f"- {x}")


if __name__ == "__main__":
    main()
