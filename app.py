import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request
import time  # 💡 引入時間模組來做動畫控制

st.markdown("""
<style>
/* 蓋住整個底部 iframe 區域（user 看到的 Hosted with Streamlit）*/
body::before {
    content: '';
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100vw;
    height: 60px;
    z-index: 99999;
    pointer-events: none;
    background-color: white;
}

/* 蓋住 admin 看到的粉紅色徽章 */
body::after {
    content: '';
    position: fixed;
    bottom: 0;
    right: 0;
    width: 25px;
    height: 48px;
    z-index: 99999;
    pointer-events: none;
    background-color: white;
}

@media (prefers-color-scheme: dark) {
    body::before { background-color: #0e1117; }
    body::after { background-color: #0e1117; }
}

[data-theme="light"] body::before { background-color: white; }
[data-theme="light"] body::after { background-color: white; }

[data-theme="dark"] body::before { background-color: #0e1117; }
[data-theme="dark"] body::after { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)



# ==========================================
# 👑 1. 設定頁面與 11 類公主標籤 (完全符合系統排序)
# ==========================================
st.set_page_config(page_title="迪士尼公主辨識器", page_icon="👑", layout="wide")

# 1. 主標題
st.markdown("# 👑 《王子的尋人任務》：落跑公主在那兒??")

# 2. 放大副標題
# st.markdown("## 🔍 昨晚跳舞，今早就認不出？讓 AI 來拯救王子的重度臉盲症")

# 3. 放大操作說明
# st.markdown("### 📸 別再拿玻璃鞋挨家挨戶蹭，讓皇家搜查官和 AI 魔鏡幫忙找出她是誰！")
st.markdown("### 🔍👠 別再拿玻璃鞋挨家挨戶蹭，讓皇家搜查官和 AI 魔鏡幫忙找出她是誰！")

# 分隔線
st.markdown("---")

# 4. 放大版的上傳提示
st.markdown("### 📥 請上傳一張迪士尼公主的圖片...")

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
# ⚙️ 2. 5-Fold 核心模型載入與下載邏輯 (全新重構)
# ==========================================
# 保持這個核心函數乾淨，負責處理單一模型的建立與下載
@st.cache_resource
def load_single_fold_model_core(fold_idx, weights_path, download_url, num_classes):
    # 建立模型架構 (與訓練時完全相同)
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # 配合優化後的訓練架構
        nn.Linear(in_features, num_classes)
    )
    
    # 檢查本地是否存在權重，若無則聯網下載
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve(download_url, weights_path)
                
    # 載入權重並強制到 CPU
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 負責迴圈處理 5 個模型下載的 UI 提示
def get_ensemble_models():
    models_list = []
    num_classes = len(CLASS_NAMES)
    
    # 這裡請換成你實際託管 5 個權重的 GitHub Release 網址
    # 範例假設你的檔名為 resnet50_fold_1_best.pth 到 resnet50_fold_5_best.pth
    base_url = "https://github.com/alohabearbear-sudo/disney-princess/releases/download/v1"
    
    try:
        # 建立一個下載進度提示
        missing_folds = [i for i in range(1, 6) if not os.path.exists(f"resnet50_fold_{i}_best.pth")]
        
        if missing_folds:
            with st.spinner(f"📥 正在從 GitHub 下載皇家審查官 5-Fold 雲端模型權重（僅需下載一次，請稍候...）"):
                for i in range(1, 6):
                    w_path = f"resnet50_fold_{i}_best.pth"
                    d_url = f"{base_url}/resnet50_fold_{i}_best.pth"
                    mdl = load_single_fold_model_core(i, w_path, d_url, num_classes)
                    models_list.append(mdl)
            st.toast("🎉 皇家審查官 5-Fold 模型權重全數加載成功！", icon="✅")
        else:
            # 如果全都在本地了，靜悄悄加載
            for i in range(1, 6):
                w_path = f"resnet50_fold_{i}_best.pth"
                d_url = f"{base_url}/resnet50_fold_{i}_best.pth"
                mdl = load_single_fold_model_core(i, w_path, d_url, num_classes)
                models_list.append(mdl)
                
        return models_list
    except Exception as e:
        st.error(f"❌ 載入 5-Fold 模型失敗！原因: {e}")
        return None

# 🚀 正式初始化載入「5個模型」組成的皇家委員會
models_ensemble = get_ensemble_models()

# ==========================================
# 🖼️ 3. 影像預處理設定
# ==========================================
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 📤 4. 前端 UI 與預測邏輯 (5-Fold 集成投票版)
# ==========================================
uploaded_file = st.file_uploader("快動手試試看吧!", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 讀取並轉換圖片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 執行 5-Fold 集成預測邏輯
    if models_ensemble is not None and len(models_ensemble) == 5:
        input_tensor = transform_pipeline(image).unsqueeze(0)
        
        fold_probabilities = []
        with torch.no_grad():
            # 讓 5 個模型各自看圖預測
            for model in models_ensemble:
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                fold_probabilities.append(probs)
        
        # 🔥【核心修改】：將 5 個模型的預測機率取平均 (Ensemble Average)
        probabilities = torch.stack(fold_probabilities).mean(dim=0)
        
        # 從平均機率中找出最大值作為最終決定
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item() * 100

    # 使用黃金比例 [4.5, 5.5] 切分左右欄位
    col1, col2 = st.columns([4.5, 5.5], gap="large")
    
    # ---- 👈 左側欄位 ----
    with col1:
        st.image(image, caption="📷 上傳的圖片", use_container_width=True)
        st.write("🧠 皇家審查官密集搜查中...")
        st.success(f"🎉 AI 魔鏡判定：昨晚舞會那位女孩是 **{predicted_class}**")
        st.info(f"📊 AI 魔鏡信心度 (Ensemble Confidence)：**{score:.2f}%**")
        
    # ---- 👉 右側欄位 ----
    with col2:
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
        
        st.write("🔍 **5-Fold 綜合評估｜所有 11 名公主候選機率排名：**")
        
        # 進行 11 名的大到小排序
        all_prob, all_idx = torch.topk(probabilities, len(CLASS_NAMES))
        
        # 動態動畫
        progress_placeholders = []
        target_probabilities = []
        
        for i in range(len(CLASS_NAMES)):
            prob_value = all_prob[i].item()      
            prob_percentage = prob_value * 100   
            class_name = CLASS_NAMES[all_idx[i].item()]
            
            st.write(f"{i+1}. {class_name}: **{prob_percentage:.2f}%**")
            ph = st.empty()
            ph.progress(0.0)
            
            progress_placeholders.append(ph)
            target_probabilities.append(prob_value)
            
        # 🚀 衝刺動畫
        steps = 25 
        for step in range(1, steps + 1):
            ratio = step / steps
            for i in range(len(CLASS_NAMES)):
                target = target_probabilities[i]
                current_val = target * ratio
                progress_placeholders[i].progress(min(current_val, 1.0))
            time.sleep(0.02)
