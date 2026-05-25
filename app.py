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
st.set_page_config(page_title="迪士尼公主 AI 辨識系統", page_icon="👑", layout="wide")

st.title("👑 AI的精準度：玻璃鞋能幫王子找對公主嗎？")
st.write("上傳一張迪士尼公主的圖片，讓玻璃鞋幫忙辨識她是誰！")

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
# 📤 4. 前端 UI 與預測邏輯 (緊湊無擋版 - 完美貼齊底部)
# ==========================================
uploaded_file = st.file_uploader("選擇一張公主圖片...", type=["jpg", "jpeg", "png", "bmp"])

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

    # 使用 st.columns 切成左右兩邊
    col1, col2 = st.columns(2)
    
    # ---- 👈 左側欄位：放上傳的圖片與主要預測結果 ----
    with col1:
        st.image(image, caption="📷 上傳的圖片", use_container_width=True)
        st.write("🧠 AI 辨識中...")
        st.success(f"🎉 辨識結果：**{predicted_class}**")
        st.info(f"📊 信心度 (Confidence)：**{score:.2f}%**")
        
    # ---- 👉 右側欄位：透過官方 st.html 微調間距，全數排開不擋到 ----
    with col2:
        # ✨ Streamlit 官方標準全域 CSS 壓縮法，把原生的間距縮小，絕對不噴原始碼
        st.html(
            """
            <style>
                /* 壓縮 Streamlit 的文字塊上下間距 */
                .stElementContainer div[data-testid="stMarkdownContainer"] p {
                    margin-bottom: 0px !important;
                    margin-top: 2px !important;
                    font-size: 13.5px !important;
                }
                /* 壓縮進度條內建的上下大空白 */
                .stElementContainer has-st-progress {
                    margin-bottom: 2px !important;
                    margin-top: 0px !important;
                }
                /* 針對舊版或通用 progress 容器微調 */
                div[data-testid="stProgress"] {
                    margin-bottom: 4px !important;
                    padding-bottom: 0px !important;
                }
            </style>
            """
        )
        
        st.write("🔍 **所有 11 名公主候選機率排名：**")
        
        # 進行 11 名的大到小排序
        all_prob, all_idx = torch.topk(probabilities, len(CLASS_NAMES))
        
        # 移除了 with st.container(height=...) 限制，讓它自然舒展，不再被截斷！
        for i in range(len(CLASS_NAMES)):
            prob_value = all_prob[i].item()      
            prob_percentage = prob_value * 100   
            class_name = CLASS_NAMES[all_idx[i].item()]
            
            # 使用原生的 write 顯示名字與百分比
            st.write(f"{i+1}. {class_name}: **{prob_percentage:.2f}%**")
            # 使用原生的 progress 進度條
            st.progress(prob_value)
