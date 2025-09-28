# 🏠 Real Estate Price Prediction with Artificial Neural Network -Description in English

A complete machine learning project for predicting residential unit prices using artificial neural networks and the TensorFlow/Keras library.

## 📁 Project Structure

```
real_estate_ann_project/          # Main project folder
│
├── data/                         # Data folder
│   ├── processed/                # Processed data
│   │   ├── X_test_encoded.csv   # Standardized test features
│   │   ├── X_train_encoded.csv  # Standardized training features
│   │   ├── X_val_encoded.csv    # Standardized validation features
│   │   ├── y_test_encoded.csv   # Standardized test target values
│   │   ├── y_train_encoded.csv  # Standardized training target values
│   │   └── y_val_encoded.csv    # Standardized validation target values
│   └── building_dataset_en_10k.csv  # Original raw dataset
│
├── plots/                        # Folder containing all charts and images
│   ├── area_price_rooms_bubble.png      # Bubble chart of area-price relationship
│   ├── faceted_price_distribution.png   # Price distribution by categorical features
│   ├── learning_curve_r2.png            # R-squared learning curve
│   └── price_distribution.png           # Price distribution of units
│
│── notebooks/  # Jupyter notebooks
├   ├── real_estate_price_prediction_ann_EN.ipynb  # Main project notebook (complete code) with English explanations and comments
├   └── real_estate_price_prediction_ann_FA.ipynb  # Main project notebook (complete code) with Persian explanations and comments
├── README.md                  # This guide file
└── requirements.txt           # Project requirements file (list of required libraries)
```

## 📊 Dataset

### General Specifications
- **Number of samples**: 10,000 records
- **Number of features**: 47 features
- **Numerical features**: 6 features
- **Categorical features**: 41 features
- **Target variable**: Price (values between 21,203 and 75,078)

### Numerical Features
1. **Floor**: Floor number (values 1 to 5)
2. **Unit_Area_sq_m**: Unit area (140 to 314 square meters)
3. **Number_of_Rooms**: Number of rooms (2, 3, or 4)
4. **Parking_Spots**: Number of parking spots (0, 1, or 2)
5. **Storage_Unit**: Storage unit availability (0 or 1)
6. **Price**: Price (target variable)

### Important Categorical Features
- **Foundation_and_Structure**: Foundation and structure type
- **Floor_Ceiling**: Floor and ceiling type
- **Wall_Materials**: Wall materials
- **Heating_System**: Heating system
- **Cooling_System**: Cooling system
- **Electrical_Wiring**: Electrical wiring
- And 36 other features related to building equipment and finishes

## 🛠️ Project Stages

### 1. Initial Data Exploration
- Dataframe structure exploration with `df.info()`
- Descriptive statistics with `df.describe()`
- Unique value analysis with `df.nunique()`
- Duplicate and missing values check

### 2. Exploratory Data Analysis (EDA)
- Analysis of unit price distribution
- Examination of price relationships with various features
- Correlation analysis between variables
- Price distribution analysis based on categorical features
- Analysis of relationship between area, number of rooms, and price

### 3. Data Preprocessing
- Splitting data into independent (X) and dependent (y) variables
- Data splitting into training (60%), validation (20%), and test (20%) sets
- Standardization of numerical features with `StandardScaler`
- Encoding categorical features with `OneHotEncoder`
- Creating preprocessing pipeline with `ColumnTransformer`

### 4. Model Design and Training
- Neural network design with two hidden layers (64 and 32 neurons)
- Using Dropout (0.1) to reduce overfitting
- Using Regularization (L2) to improve generalization
- Model compilation with Adam optimizer and MSE loss function
- Using MAE, MSE, and R-squared metrics for evaluation
- Model training with 100 epochs and batch size 32
- Using callbacks for Early Stopping and learning rate reduction

### 5. Model Evaluation
- Model evaluation on training, validation, and test data
- Calculation of various evaluation metrics
- Returning predictions to original scale
- Comparison of predictions with actual values

### 6. Results Visualization
- Price distribution chart
- Faceted charts for categorical features
- Bubble plot for multivariate relationships
- Learning curve and R-squared improvement trend

## 📈 Model Results

### Final Model Performance:

| Dataset        | Loss (MSE) | MAE    | R-squared |
| -------------- | ---------- | ------ | --------- |
| **Training**   | 0.0239     | 0.0962 | 0.9850    |
| **Validation** | 0.0298     | 0.1152 | 0.9773    |
| **Test**       | 0.0280     | 0.1105 | 0.9800    |

### Results Analysis:
- The model achieved very high accuracy (R² > 0.98) in price prediction
- Small difference between training and validation error indicates no overfitting
- Excellent performance on test data shows model generalization capability

## 🔧 Techniques Used

### Neural Network Architecture:
```python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 64)             │         8,192 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

### Training Settings:
- **Loss function**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Evaluation metrics**: MAE, MSE, R²
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.2, patience=5)

## 🚀 Execution Guide

1. **Create virtual environment** (optional but recommended):
```bash
python -m venv myvenv
source myvenv/bin/activate  # For Linux/Mac
myvenv\Scripts\activate     # For Windows
```

2. **Install requirements**:
```bash
pip install -r requirements.txt
```

3. **Run the project**:
   - Open `real_estate_price_prediction_ann.ipynb` in Jupyter Notebook or Jupyter Lab
   - Execute cells in order

4. **View results**:
   - Training results are displayed during the process
   - Charts are saved in the `plots` folder
   - Final model evaluation is displayed at the end of the notebook

## 📝 Additional Information

### About the Data:
This dataset includes 10,000 samples of residential units with 47 different features that provide complete details about technical specifications, equipment, and materials used in the buildings. Features are divided into different categories:

1. **Basic unit specifications**: Floor, area, number of rooms, parking, storage
2. **Structure and facade specifications**: Foundation type, wall materials, facade material
3. **Equipment and finishes**: Flooring, paint, doors and windows, cabinets
4. **Installations**: Heating/cooling systems, electrical wiring, plumbing
5. **Additional amenities**: Lighting, landscaping, intercom, door opener

### Project Applications:
- Predicting real estate prices based on technical features
- Analyzing the impact of various features on unit prices
- Providing a model for property valuation by construction companies
- Serving as a foundation for real estate recommendation systems

---
👋 We hope you find this project useful! 🚀

##  👤 Developer Contact 
    Email: masudpythongit@gmail.com 
    Telegram: https://t.me/Masoud_Ghasemi_sorna_fast
🔗 GitHub Account: [sorna-fast](https://github.com/sorna-fast)

---

## 📄 License

This project is released under the [MIT](LICENSE) license.

---
---

# 🏠 پیش‌بینی قیمت املاک با شبکه عصبی مصنوعی - Description in Persian

یک پروژه کامل یادگیری ماشین برای پیش‌بینی قیمت واحدهای مسکونی با استفاده از شبکه عصبی مصنوعی و کتابخانه TensorFlow/Keras.

## 📁 ساختار پروژه

```
real_estate_ann_project/          # پوشه اصلی پروژه
│
├── data/                         # پوشه مربوط به داده‌ها
│   ├── processed/                # داده‌های پردازش شده
│   │   ├── X_test_encoded.csv   # ویژگی‌های تست استاندارد شده
│   │   ├── X_train_encoded.csv  # ویژگی‌های آموزش استاندارد شده
│   │   ├── X_val_encoded.csv    # ویژگی‌های اعتبارسنجی استاندارد شده
│   │   ├── y_test_encoded.csv   # مقادیر هدف تست استاندارد شده
│   │   ├── y_train_encoded.csv  # مقادیر هدف آموزش استاندارد شده
│   │   └── y_val_encoded.csv    # مقادیر هدف اعتبارسنجی استاندارد شده
│   └── building_dataset_en_10k.csv  # دیتاست اصلی و خام
│
├── plots/                        # پوشه حاوی تمام نمودارها و تصاویر
│   ├── area_price_rooms_bubble.png      # نمودار حبابی رابطه مساحت و قیمت
│   ├── faceted_price_distribution.png   # توزیع قیمت براساس ویژگی‌های دسته‌ای
│   ├── learning_curve_r2.png            # منحنی یادگیری R-squared
│   └── price_distribution.png           # توزیع قیمت واحدها
│
├── notebooks/                         # ژوپیتر نوت بوک
│   ├── real_estate_price_prediction_ann_EN.ipynb  # نوت‌بوک اصلی پروژه (کد کامل) توضیح و کامنت به زبان انگلیسی
│   └── real_estate_price_prediction_ann_FA.ipynb  #  نوت‌بوک اصلی پروژه (کد کامل) توضیح و کامنت به زبان فارسی
├── README.md                  # این فایل راهنما
└── requirements.txt           # فایل نیازمندی‌های پروژه (لیست کتابخانه‌های مورد نیاز)
```

## 📊 مجموعه داده

### مشخصات کلی
- **تعداد نمونه**: ۱۰,۰۰۰ رکورد
- **تعداد ویژگی‌ها**: ۴۷ ویژگی
- **تعداد ویژگی‌های عددی**: ۶ ویژگی
- **تعداد ویژگی‌های دسته‌ای**: ۴۱ ویژگی
- **متغیر هدف**: قیمت (Price) - مقادیر بین 21,203 تا 75,078

### ویژگی‌های عددی
1. **Floor**: طبقه (مقادیر 1 تا 5)
2. **Unit_Area_sq_m**: مساحت واحد (140 تا 314 متر مربع)
3. **Number_of_Rooms**: تعداد اتاق (2، 3 یا 4)
4. **Parking_Spots**: تعداد جای پارک (0، 1 یا 2)
5. **Storage_Unit**: وجود انباری (0 یا 1)
6. **Price**: قیمت (متغیر هدف)

### ویژگی‌های دسته‌ای مهم
- **Foundation_and_Structure**: نوع فونداسیون و سازه
- **Floor_Ceiling**: نوع کف و سقف
- **Wall_Materials**: مصالح دیوارها
- **Heating_System**: سیستم گرمایشی
- **Cooling_System**: سیستم سرمایشی
- **Electrical_Wiring**: سیم‌کشی برق
- و 36 ویژگی دیگر مربوط به تجهیزات و finishes ساختمان

## 🛠️ مراحل پروژه

### 1. بررسی اولیه داده‌ها
- بررسی ساختار دیتافریم با `df.info()`
- بررسی آمار توصیفی با `df.describe()`
- بررسی مقادیر یکتا با `df.nunique()`
- بررسی مقادیر تکراری و missing values

### 2. آنالیز اکسپلوراتوری داده (EDA)
- تحلیل توزیع قیمت واحدها
- بررسی رابطه قیمت با ویژگی‌های مختلف
- تحلیل همبستگی بین متغیرها
- بررسی توزیع قیمت براساس ویژگی‌های دسته‌ای
- تحلیل رابطه بین مساحت، تعداد اتاق و قیمت

### 3. پیش‌پردازش داده‌ها
- تقسیم داده به متغیرهای مستقل (X) و وابسته (y)
- تقسیم داده به مجموعه‌های آموزش (60%)، اعتبارسنجی (20%) و آزمون (20%)
- استانداردسازی ویژگی‌های عددی با `StandardScaler`
- کدگذاری ویژگی‌های دسته‌ای با `OneHotEncoder`
- ایجاد pipeline پیش‌پردازش با `ColumnTransformer`

### 4. طراحی و آموزش مدل
- طراحی شبکه عصبی با دو لایه پنهان (64 و 32 نرون)
- استفاده از Dropout (0.1) برای کاهش overfitting
- استفاده از Regularization (L2) برای بهبود تعمیم‌پذیری
- کامپایل مدل با optimizer Adam و loss function MSE
- استفاده از معیارهای MAE, MSE و R-squared برای ارزیابی
- آموزش مدل با 100 epoch و batch size 32
- استفاده از callbacks برای Early Stopping و کاهش نرخ یادگیری

### 5. ارزیابی مدل
- ارزیابی مدل روی داده‌های آموزش، اعتبارسنجی و آزمون
- محاسبه معیارهای مختلف ارزیابی
- بازگرداندن پیش‌بینی‌ها به مقیاس اصلی
- مقایسه پیش‌بینی‌ها با مقادیر واقعی

### 6. تجسم نتایج
- رسم نمودار توزیع قیمت
- رسم نمودارهای facet شده برای ویژگی‌های دسته‌ای
- رسم bubble plot برای رابطه چندمتغیره
- رسم منحنی یادگیری و روند بهبود R-squared

## 📈 نتایج مدل

### عملکرد نهایی مدل:

| مجموعه داده    | Loss (MSE) | MAE    | R-squared |
| -------------- | ---------- | ------ | --------- |
| **آموزش**      | 0.0239     | 0.0962 | 0.9850    |
| **اعتبارسنجی** | 0.0298     | 0.1152 | 0.9773    |
| **آزمون**      | 0.0280     | 0.1105 | 0.9800    |

### تحلیل نتایج:
- مدل به دقت بسیار بالایی (R² > 0.98) در پیش‌بینی قیمت دست یافته است
- اختلاف کم بین خطای آموزش و اعتبارسنجی نشان‌دهنده عدم overfitting است
- عملکرد عالی روی داده آزمون نشان‌دهنده قابلیت تعمیم‌پذیری مدل است

## 🔧 تکنیک‌های استفاده شده

### معماری شبکه عصبی:
```python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 64)             │         8,192 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

### تنظیمات آموزش:
- **تابع loss**: MSE (میانگین مربعات خطا)
- **بهینه‌ساز**: Adam
- **معیارهای ارزیابی**: MAE, MSE, R²
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.2, patience=5)


## 🚀 راهنمای اجرا

1. **ایجاد محیط مجازی** (اختیاری اما توصیه شده):
```bash
python -m venv myvenv
source myvenv/bin/activate  # برای Linux/Mac
myvenv\Scripts\activate     # برای Windows
```

2. **نصب نیازمندی‌ها**:
```bash
pip install -r requirements.txt
```

3. **اجرای پروژه**:
   - فایل `real_estate_price_prediction_ann.ipynb` را در Jupyter Notebook یا Jupyter Lab باز کنید
   - سلول‌ها را به ترتیب اجرا کنید

4. **مشاهده نتایج**:
   - نتایج آموزش در طول فرآیند نمایش داده می‌شود
   - نمودارها در پوشه `plots` ذخیره می‌شوند
   - ارزیابی نهایی مدل در انتهای نوت‌بوک نمایش داده می‌شود

## 📝 توضیحات تکمیلی

### درباره داده‌ها:
این مجموعه داده شامل ۱۰,۰۰۰ نمونه از واحدهای مسکونی با ۴۷ ویژگی مختلف است که جزئیات کاملی از مشخصات فنی، تجهیزات و مصالح به کار رفته در ساختمان‌ها را ارائه می‌دهد. ویژگی‌ها به دسته‌های مختلفی تقسیم می‌شوند:

1. **مشخصات اصلی واحد**: طبقه، مساحت، تعداد اتاق، پارکینگ، انباری
2. **مشخصات سازه و نما**: نوع فونداسیون، مصالح دیوارها، جنس نما
3. **تجهیزات و finishes**: کفپوش، رنگ، درب و پنجره، کابینت
4. **تاسیسات**: سیستم‌های گرمایشی/سرمایشی، سیم‌کشی برق، لوله‌کشی
5. **امکانات جانبی**: روشنایی، محوطه‌سازی، آیفون، درب بازکن

### کاربردهای پروژه:
- پیش‌بینی قیمت املاک بر اساس ویژگی‌های فنی
- تحلیل تاثیر ویژگی‌های مختلف بر قیمت واحدها
- ارائه مدلی برای برآورد ارزش ملک توسط شرکت‌های ساختمانی
- استفاده به عنوان پایه‌ای برای سیستم‌های توصیه‌گر املاک


---
👋 امیدواریم این پروژه برای شما مفید باشد! 🚀

##  👤 ارتباط با توسعه‌دهنده 
    ایمیل: masudpythongit@gmail.com 
    تلگرام: https://t.me/Masoud_Ghasemi_sorna_fast
🔗 حساب گیتهاب: [sorna-fast](https://github.com/sorna-fast)

---


## 📄 مجوز

این پروژه تحت مجوز [MIT](LICENSE) منتشر شده است.

