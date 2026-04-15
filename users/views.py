from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter issues in Django
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import google.generativeai as genai
from django.conf import settings

# Your Gemini API Key
genai.configure(api_key="AIzaSyAysVqWZ-Ydq8NTcPZN6QwpVX5JkEDE17Q")

# Base Page
def base(request):
    return render(request, 'base.html')

# Registration
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful.')
            return render(request, 'UserRegistration.html')
        else:
            messages.error(request, ' Email or Mobile already exists.')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistration.html', {'form': form})

# Login
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if user.status == "activated":
                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                return redirect('UserHome')
            else:
                messages.error(request, 'Your account is not activated.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID or Password')
    return render(request, 'UserLogin.html')

# Dashboard
def UserHome(request):
    return render(request, 'users/UserHome.html')

#  AI-Powered Dataset Analysis
def analyse_dataset(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save('last_uploaded.csv', uploaded_file)
        filepath = fs.path(filename)

        try:
            # File type checks
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(filepath, encoding='utf-8', engine='python')
            elif ext == '.tsv':
                df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            elif ext == '.xlsx':
                df = pd.read_excel(filepath, engine='openpyxl')
            elif ext == '.xls':
                df = pd.read_excel(filepath, engine='xlrd')
            else:
                raise ValueError('Unsupported file format. Please upload a CSV, Excel, or TSV file.')

            context['columns'] = df.columns.tolist()
            context['shape'] = df.shape
            context['describe_html'] = df.describe(include='all').fillna('').to_html(classes='table table-bordered')

            # Data Quality Metrics
            numeric_cols_list = df.select_dtypes(include='number').columns.tolist()
            categorical_cols_list = df.select_dtypes(include='object').columns.tolist()
            
            # Check for null values
            total_null_values = df.isnull().sum().sum()
            null_percentage = (total_null_values / (len(df) * len(df.columns))) * 100
            has_nulls = total_null_values > 0
            
            # Check for data imbalance
            imbalance_info = {}
            for col in categorical_cols_list:
                if df[col].nunique() <= 20:  # Only check columns with reasonable number of classes
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 1:
                        max_count = value_counts.max()
                        min_count = value_counts.min()
                        imbalance_ratio = max_count / min_count if min_count > 0 else 0
                        imbalance_info[col] = {
                            'counts': value_counts.to_dict(),
                            'ratio': imbalance_ratio,
                            'is_balanced': imbalance_ratio <= 1.5
                        }
            
            # Determine if data is balanced overall
            is_balanced = all(info['is_balanced'] for info in imbalance_info.values()) if imbalance_info else True
            
            # Add to context
            context['data_quality'] = {
                'has_nulls': has_nulls,
                'null_count': int(total_null_values),
                'null_percentage': round(null_percentage, 2),
                'numeric_cols': numeric_cols_list,
                'categorical_cols': categorical_cols_list,
                'is_balanced': is_balanced,
                'imbalance_info': imbalance_info
            }
            
            # Generate model recommendations based on target variable
            model_recommendations = "For Classification: Logistic Regression, Random Forest, SVM, Gradient Boosting\n"
            model_recommendations += "For Regression: Linear Regression, Decision Trees, XGBoost\n"
            model_recommendations += "For Clustering: K-Means, DBSCAN, Hierarchical Clustering"
            context['model_recommendations'] = model_recommendations

            # Save most frequent categorical column name to session
            cat_cols = df.select_dtypes(include='object').columns
            most_common_col = None
            for col in cat_cols:
                if df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.9:
                    most_common_col = col
                    break
            request.session['dashboard_col'] = most_common_col

            # Correlation Heatmap - Improved Size and Quality
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                corr_matrix = df[numeric_cols].corr()
                fig_size = min(max(8, len(numeric_cols) * 0.8), 14)
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           ax=ax, cbar_kws={'label': 'Correlation'}, 
                           linewidths=0.5, linecolor='gray', square=True)
                ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
                image_stream = io.BytesIO()
                plt.tight_layout()
                fig.savefig(image_stream, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                image_stream.seek(0)
                heatmap_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
                context['heatmap'] = heatmap_base64
            else:
                context['heatmap'] = None

            # Gemini AI Insight - Enhanced with Dataset Summary & ML Recommendations
            # Get dataset statistics
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            missing_data = df.isnull().sum().sum()
            total_missing_pct = (missing_data / (len(df) * len(df.columns))) * 100
            
            # Create comprehensive analysis prompt
            sample_data = df.head(10).to_csv(index=False)
            dataset_info = f"""
Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
Numeric Columns: {len(numeric_cols)} - {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
Categorical Columns: {len(categorical_cols)} - {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
Missing Values: {missing_data} ({total_missing_pct:.2f}%)

Data Sample:
{sample_data}
"""
            
            prompt = f"""You are an expert ML engineer and data scientist. Analyze this dataset and provide CONCISE information in this exact format:

📊 DATASET SUMMARY (2-3 sentences max):
[Brief overview of what the dataset contains, its size, and main characteristics]

🔧 DATA QUALITY ISSUES:
[List 2-3 key issues if any, or "No major issues detected"]

🎯 POSSIBLE TASKS & RECOMMENDED ML MODELS:

For Classification Tasks:
- [Task]: Model recommendations (explain why)

For Regression Tasks:
- [Task]: Model recommendations (explain why)

For Clustering/Analysis Tasks:
- [Task]: Model recommendations (explain why)

⚠️ PREPROCESSING NEEDED:
[List 2-3 important preprocessing steps if needed]

Dataset Info:
{dataset_info}
"""
            
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                temperature=0.5, top_p=1, max_output_tokens=1024))
            context['ai_insight'] = response.text if hasattr(response, 'text') else str(response)

        except Exception as e:
            messages.error(request, f" Error analyzing dataset: {e}")

    return render(request, 'users/analyse.html', context)

#  Dashboard View
def dashboard_view(request):
    df_path = os.path.join(settings.MEDIA_ROOT, 'last_uploaded.csv')
    if not os.path.exists(df_path):
        messages.error(request, "No dataset available. Please analyze one first.")
        return redirect('generate_docs')

    df = pd.read_csv(df_path)
    charts = []

    # Get all categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Limit to 5 charts for performance
    selected_cols = (cat_cols + num_cols)[:5]

    for col in selected_cols:
        if col in df.columns and df[col].nunique() > 1:
            chart_data = df[col].value_counts().reset_index()
            chart_data.columns = ['label', 'value']
            fig = px.pie(chart_data, names='label', values='value',
                         title=f'{col} Distribution', hole=0.3)
            fig.update_traces(textinfo='percent+label')
            charts.append(json.dumps(fig, cls=PlotlyJSONEncoder))

    return render(request, 'users/dashboard.html', {
        'charts': charts,
        'back_url': 'generate_docs'
    })




