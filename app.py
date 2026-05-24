import streamlit as st
import tempfile, os
from PIL import Image
from fabric_system import generate

st.set_page_config(
    page_title="Fabric Pattern System",
    page_icon="🧵",
    layout="wide"
)

st.title("🧵 Fabric Pattern System")
st.caption("Upload any image → get a cross-stitch chart, thread coordinates, and tiled fabric preview")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pattern Settings")

    grid_w = st.slider("Grid width (cells)", 10, 80, 40, step=5)
    grid_h = st.slider("Grid height (cells)", 10, 80, 40, step=5)
    n_colors = st.slider("Thread colors", 2, 20, 8)
    bg_option = st.selectbox("Background to remove",
                             ["white", "none", "custom"])
    if bg_option == "custom":
        r = st.number_input("R", 0, 255, 255)
        g = st.number_input("G", 0, 255, 255)
        b = st.number_input("B", 0, 255, 255)
        bg_color = f"{int(r)},{int(g)},{int(b)}"
    else:
        bg_color = bg_option

    gap = st.slider("Tile gap (cells)", 0, 5, 1)
    title = st.text_input("Chart title (optional)", "")

# ── Main panel ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your image (PNG, JPG, WEBP)",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Input Image")
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        st.caption(f"Original size: {img.width} × {img.height} px")

    if st.button("Generate Fabric Pattern", type="primary", use_container_width=True):
        with st.spinner("Processing image..."):
            # Save upload to temp file
            suffix = os.path.splitext(uploaded.name)[-1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            prefix = os.path.join(tempfile.gettempdir(),
                                  os.path.splitext(uploaded.name)[0] + "_fabric")
            try:
                result = generate(
                    image_path    = tmp_path,
                    grid_w        = grid_w,
                    grid_h        = grid_h,
                    n_colors      = n_colors,
                    bg_color      = bg_color,
                    output_prefix = prefix,
                    gap           = gap,
                    title         = title or uploaded.name,
                )
                st.success(f"Done! Grid: {grid_w}×{grid_h}  |  Colors: {n_colors}")

                # ── Show outputs ──
                st.subheader("Cross-Stitch Chart")
                st.image(result['chart'], use_container_width=True)
                with open(result['chart'], 'rb') as f:
                    st.download_button("Download chart PNG", f, file_name="fabric_chart.png",
                                       mime="image/png", use_container_width=True)

                st.subheader("Tiled Fabric Preview")
                st.image(result['preview'], use_container_width=True)
                with open(result['preview'], 'rb') as f:
                    st.download_button("Download preview PNG", f, file_name="fabric_preview.png",
                                       mime="image/png", use_container_width=True)

                st.subheader("Thread Coordinates")
                with open(result['coordinates'], 'r') as f:
                    coords_text = f.read()
                st.text_area("", coords_text, height=300)
                st.download_button("Download coordinates TXT", coords_text,
                                   file_name="fabric_coordinates.txt",
                                   mime="text/plain", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)
else:
    st.info("Upload an image on the left to get started. Try a drawing, photo, logo, or pixel art.")

    st.subheader("Demo Output")
    demo_chart = "demo_cat_fabric_chart.png"
    demo_preview = "demo_cat_fabric_preview.png"
    if os.path.exists(demo_chart):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Demo chart — Siamese cat (30×36 grid, 7 DMC colors)")
            st.image(demo_chart, use_container_width=True)
        with c2:
            st.caption("Demo tiled fabric preview")
            st.image(demo_preview, use_container_width=True)

st.divider()
st.caption("Fabric Pattern System · Each X = one cross-stitch · Coordinates: column-letter + row-number (A01, B12...)")
