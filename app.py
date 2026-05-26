import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# ==========================================
# 👑 1. 設定頁面與 11 類公主標籤 (完全符合系統排序)
# ==========================================
st.set_page_config(page_title="迪士尼公主辨識器", page_icon="👑", layout="wide")

import streamlit as st

st.markdown("# 👑 《王子的尋人啟事》：")

# 第二行：改為 24px
st.markdown('<span style="font-size: 24px;">🔍 昨晚跳舞今早認不出？讓 AI 來拯救王子的重度臉盲症 by Jimmy Chen</span>', unsafe_allow_html=True)

# 第三行：改為 20px
st.markdown('<span style="font-size: 20px;">📸 別再拿玻璃鞋挨家挨戶蹭，請上傳一張迪士尼公主的圖片，讓 AI 皇家搜查官幫忙辨識她是誰！</span>', unsafe_allow_html=True)

# 第四行：改為 20px，並留空 file_uploader 的 label
st.markdown('<span style="font-size: 20px; font-weight: bold;">📥 請選擇一張迪士尼公主的圖片...</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# ⚠️ 系統中大寫排前面、小寫排後面的絕對順序
CLASS_NAMES = [
    "Cinderella (灰姑娘)", 
    "Snow White (白雪公主)", 
    "Anna (安娜)", 
    "Ariel (小美人魚)", 
    "Aurora (睡美人/arura)", 
    "Belle (貝兒)", 
    "Elsa (艾莎)", 
    "Jasmine (茉莉公主)", 
    "Merida (梅莉達)", 
    "Rapunzel (樂佩/ruponzel)", 
    "Tiana (蒂安娜)"
]

# ==========================================
# ⚙️ 2. 載入模型權重 (修正快取 UI 衝突問題)
# ==========================================
# 保持這個函數乾淨，裡面絕對不放任何 st.xxx 元件
@st.cache_resource
def load_princess_model_core(weights_path, download_url, num_classes):
    # 建立與訓練時完全相同的 ResNet-50 架構
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 檢查本地是否存在權重，若無則聯網下載
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve(download_url, weights_path)
                
    # 載入權重 (強制對應到 CPU)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 這個包裝函數放在外面，負責處理 UI 動畫與錯誤提示
def get_model():
    weights_path = "resnet50_fold_1_best.pth"
    download_url = "https://github.com/alohabearbear-sudo/disney-princess/releases/download/v1/resnet50_fold_1_best.pth"
    num_classes = len(CLASS_NAMES)
    
    try:
        # 如果檔案不存在，我們在快取函數外面秀出下載轉圈圈
        if not os.path.exists(weights_path):
            with st.spinner("📥 正在從 GitHub 下載雲端模型權重（僅需下載一次，請稍候...）"):
                model = load_princess_model_core(weights_path, download_url, num_classes)
            st.toast("🎉 模型權重下載成功！", icon="✅")
        else:
            # 如果檔案早就存在，直接靜悄悄地載入快取
            model = load_princess_model_core(weights_path, download_url, num_classes)
        return model
    except Exception as e:
        st.error(f"❌ 載入模型失敗！原因: {e}")
        return None

# 🚀 正式初始化載入模型
model = get_model()

# ==========================================
# 🖼️ 3. 影像預處理設定
# ==========================================
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 📤 4. 前端 UI 與預測邏輯 (動態一飛衝天版)
# ==========================================
import time  # 💡 引入時間模組來做動畫控制

uploaded_file = st.file_uploader("請選擇一張迪士尼公主的圖片...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 讀取並轉換圖片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 先跑 AI 預測邏輯
    if model is not None:
        input_tensor = transform_pipeline(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item() * 100

    # 使用黃金比例 [4.5, 5.5] 切分左右欄位
    col1, col2 = st.columns([4.5, 5.5], gap="large")
    
    # ---- 👈 左側欄位：放上傳的圖片與主要預測結果 ----
    with col1:
        st.image(image, caption="📷 上傳的圖片", use_container_width=True)
        st.write("🧠 AI 辨識中...")
        st.success(f"🎉 辨識結果：**{predicted_class}**")
        st.info(f"📊 信心度 (Confidence)：**{score:.2f}%**")
        
    # ---- 👉 右側欄位：動態進度條，一飛衝天 ----
    with col2:
        # Streamlit 官方標準全域 CSS 壓縮間距
        st.html(
            """
            <style>
                .stElementContainer div[data-testid="stMarkdownContainer"] p {
                    margin-bottom: 0px !important;
                    margin-top: 3px !important;
                    font-size: 14px !important;
                }
                .stElementContainer has-st-progress {
                    margin-bottom: 2px !important;
                    margin-top: 0px !important;
                }
                div[data-testid="stProgress"] {
                    margin-bottom: 5px !important;
                    padding-bottom: 0px !important;
                }
            </style>
            """
        )
        
        st.write("🔍 **所有 11 名公主候選機率排名：**")
        
        # 進行 11 名的大到小排序
        all_prob, all_idx = torch.topk(probabilities, len(CLASS_NAMES))
        
        # 💡 關鍵動態魔法：先建立 11 個空進度條的「錨點（Placeholders）」
        progress_placeholders = []
        target_probabilities = []
        
        for i in range(len(CLASS_NAMES)):
            prob_value = all_prob[i].item()      
            prob_percentage = prob_value * 100   
            class_name = CLASS_NAMES[all_idx[i].item()]
            
            # 先渲染文字
            st.write(f"{i+1}. {class_name}: **{prob_percentage:.2f}%**")
            # 在文字下方建立一個空的、可變動的容器，並先預填 0% 的進度條
            ph = st.empty()
            ph.progress(0.0)
            
            progress_placeholders.append(ph)
            target_probabilities.append(prob_value)
            
        # 🚀 衝刺動畫迴圈：讓所有進度條同步從 0 開始往右爬升
        # 總共切成 25 步，大約在 0.5 秒內完成動畫
        steps = 25 
        for step in range(1, steps + 1):
            ratio = step / steps # 目前進度的比例 (0.04 -> 1.0)
            
            for i in range(len(CLASS_NAMES)):
                target = target_probabilities[i]
                
                # 讓動畫呈現非線性：機率低的很快就停住，機率高（接近1）的會有一路全速向上衝刺的超車感
                current_val = target * ratio
                
                # 更新對應位置的進度條
                progress_placeholders[i].progress(min(current_val, 1.0))
                
            # 每前進一步，稍微停頓一下下（微秒級），製造肉眼可見的流暢滑動感
            time.sleep(0.02)
