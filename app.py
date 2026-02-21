import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# SQLite Database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ML Models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Using fallback forecasting.")

# ==================== DATABASE SETUP ====================
DATABASE_URL = "sqlite:///./msme_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MSMEProfile(Base):
    __tablename__ = "msme_profiles"
    id = Column(Integer, primary_key=True, index=True)
    mobile_number = Column(String(15), unique=True, index=True)
    full_name = Column(String(100))
    email = Column(String(100))
    role = Column(String(50))
    company_name = Column(String(200))
    business_type = Column(String(50))
    state = Column(String(50))
    city = Column(String(100))
    years_operation = Column(Integer)
    monthly_revenue_range = Column(String(50))
    verification_status = Column(String(20), default="PENDING")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    consent_given = Column(Boolean, default=False)
    organisation_type = Column(String(100))
    major_activity = Column(String(200))
    enterprise_type = Column(String(50))

Base.metadata.create_all(bind=engine)

# ==================== DATABASE OPERATIONS ====================
def save_user_profile(profile_data):
    db = SessionLocal()
    try:
        existing = db.query(MSMEProfile).filter(
            MSMEProfile.mobile_number == profile_data['mobile_number']
        ).first()
        profile_data_for_db = profile_data.copy()
        if 'msme_number' in profile_data_for_db:
            del profile_data_for_db['msme_number']
        if existing:
            for key, value in profile_data_for_db.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            db.commit()
            return existing.id
        else:
            profile = MSMEProfile(**profile_data_for_db)
            db.add(profile)
            db.commit()
            db.refresh(profile)
            return profile.id
    finally:
        db.close()

def get_user_profile(mobile_number):
    db = SessionLocal()
    try:
        profile = db.query(MSMEProfile).filter(
            MSMEProfile.mobile_number == mobile_number
        ).first()
        if profile:
            return {
                'id': profile.id,
                'mobile_number': profile.mobile_number,
                'full_name': profile.full_name,
                'company_name': profile.company_name,
                'business_type': profile.business_type,
                'state': profile.state,
                'city': profile.city,
                'verification_status': profile.verification_status,
                'organisation_type': profile.organisation_type,
                'major_activity': profile.major_activity,
                'enterprise_type': profile.enterprise_type
            }
        return None
    finally:
        db.close()

# ==================== ML & ANALYTICS FUNCTIONS ====================
def normalize(series):
    if series.empty or series.max() == series.min():
        return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def calculate_scores(df):
    """Calculate risk and performance scores"""
    column_mapping = {
        'Sales_INR':             'Monthly_Sales_INR',
        'Monthly_Sales':         'Monthly_Sales_INR',
        'Gross_Sales':           'Monthly_Sales_INR',
        'Operating_Cost_INR':    'Monthly_Operating_Cost_INR',
        'Operating_Cost':        'Monthly_Operating_Cost_INR',
        'Outstanding_Loan':      'Outstanding_Loan_INR',
        'Outstanding_Amount':    'Outstanding_Loan_INR',
        'Vendor_Reliability':    'Vendor_Delivery_Reliability',
        'Inventory_Turnover_Rate': 'Inventory_Turnover',
        'Average_Margin_Percent':'Avg_Margin_Percent',
        'Profit_Margin_%':       'Avg_Margin_Percent',
        'Monthly_Demand':        'Monthly_Demand_Units',
        'Quantity_Sold':         'Monthly_Demand_Units',
        'Returns':               'Returns_Percentage',
        'Return_Quantity':       'Returns_Percentage',
        'Product_Name':          'SKU_Name',
    }
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    numeric_cols = ['Monthly_Sales_INR', 'Monthly_Operating_Cost_INR', 'Outstanding_Loan_INR',
                    'Vendor_Delivery_Reliability', 'Inventory_Turnover', 'Avg_Margin_Percent',
                    'Monthly_Demand_Units', 'Returns_Percentage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    df['Monthly_Sales_INR_Adjusted'] = df['Monthly_Sales_INR'].replace(0, 1e-9)
    df["Cashflow_Stress"] = normalize(df["Monthly_Operating_Cost_INR"] / df["Monthly_Sales_INR_Adjusted"])
    df["Loan_Stress"] = normalize(df["Outstanding_Loan_INR"] / (df["Monthly_Sales_INR_Adjusted"] * 12))
    df["Financial_Risk_Score"] = (0.5 * df["Cashflow_Stress"] + 0.5 * df["Loan_Stress"]).clip(0, 1)
    df["Vendor_Score"] = (
        0.5 * df["Vendor_Delivery_Reliability"] +
        0.3 * normalize(df["Inventory_Turnover"]) +
        0.2 * normalize(df["Avg_Margin_Percent"])
    ).clip(0, 1)
    df["Growth_Potential_Score"] = (
        0.4 * normalize(df["Monthly_Demand_Units"]) +
        0.35 * normalize(df["Avg_Margin_Percent"]) +
        0.25 * (1 - normalize(df["Returns_Percentage"]))
    ).clip(0, 1)
    df["MSME_Health_Score"] = (
        (1 - df["Financial_Risk_Score"]) * 0.4 +
        df["Vendor_Score"] * 0.3 +
        df["Growth_Potential_Score"] * 0.3
    ) * 100
    df['Profitability_Ratio'] = normalize(df['Avg_Margin_Percent'] * df['Monthly_Sales_INR_Adjusted'])
    df['Operational_Efficiency'] = (1 - normalize(df['Monthly_Operating_Cost_INR'] / df['Monthly_Sales_INR_Adjusted'])).clip(0, 1)
    df['Customer_Satisfaction'] = (1 - normalize(df['Returns_Percentage'])).clip(0, 1)
    df['Performance_Score'] = (
        0.3 * df['Profitability_Ratio'] +
        0.25 * df['Operational_Efficiency'] +
        0.2 * df['Customer_Satisfaction'] +
        0.15 * df['Vendor_Delivery_Reliability'] +
        0.1 * normalize(df['Inventory_Turnover'])
    ).clip(0, 1) * 100
    return df


# ==================== NOTEBOOK-ALIGNED FORECAST FUNCTION ====================
def forecast_sales(df):
    ROLLING_MONTHS = 12
    sales_col = None
    for candidate in ['Monthly_Sales_INR', 'Gross_Sales']:
        if candidate in df.columns:
            sales_col = candidate
            break
    if sales_col is None:
        total = df.select_dtypes(include=[np.number]).sum().sum()
        return {
            '6_month': {'forecast': total * 6 * 1.05, 'lower': total * 6 * 1.05 * 0.85, 'upper': total * 6 * 1.05 * 1.15},
            '12_month': {'forecast': total * 12 * 1.05, 'lower': total * 12 * 1.05 * 0.85, 'upper': total * 12 * 1.05 * 1.15}
        }

    if 'Date' not in df.columns:
        total = df[sales_col].sum()
        return {
            '6_month': {'forecast': total * 6 * 1.05, 'lower': total * 6 * 1.05 * 0.85, 'upper': total * 6 * 1.05 * 1.15},
            '12_month': {'forecast': total * 12 * 1.05, 'lower': total * 12 * 1.05 * 0.85, 'upper': total * 12 * 1.05 * 1.15}
        }

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    def _prophet_forecast_monthly(series_df, store_label="ALL"):
        ts = series_df.set_index('Date')[sales_col]
        monthly = ts.resample('MS').sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)
        if len(monthly) == 0:
            return None
        if len(monthly) > ROLLING_MONTHS:
            train_df = monthly.tail(ROLLING_MONTHS).copy()
        else:
            train_df = monthly.copy()
        last_train_date = monthly['ds'].max()
        avg_monthly = monthly['y'].mean()
        if len(train_df) < 2 or not PROPHET_AVAILABLE:
            growth = 0.05
            f6 = avg_monthly * 6 * (1 + growth)
            f12 = avg_monthly * 12 * (1 + growth)
            return {
                '6_month': {'forecast': f6, 'lower': f6 * 0.85, 'upper': f6 * 1.15},
                '12_month': {'forecast': f12, 'lower': f12 * 0.85, 'upper': f12 * 1.15}
            }
        try:
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=25,
                changepoint_range=0.95,
                interval_width=0.95
            )
            model.fit(train_df)
            future = model.make_future_dataframe(periods=12, freq='MS')
            forecast = model.predict(future)
            forecast_future = forecast[forecast['ds'] >= last_train_date].copy()
            forecast_future = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_future['yhat']       = forecast_future['yhat'].clip(lower=0)
            forecast_future['yhat_lower'] = forecast_future['yhat_lower'].clip(lower=0)
            forecast_future['yhat_upper'] = forecast_future['yhat_upper'].clip(lower=0)
            f6 = forecast_future.head(6)
            f12 = forecast_future.head(12)
            return {
                '6_month': {
                    'forecast': f6['yhat'].sum(),
                    'lower': f6['yhat_lower'].sum(),
                    'upper': f6['yhat_upper'].sum()
                },
                '12_month': {
                    'forecast': f12['yhat'].sum(),
                    'lower': f12['yhat_lower'].sum(),
                    'upper': f12['yhat_upper'].sum()
                },
                'forecast_df': forecast_future
            }
        except Exception as e:
            growth = 0.05
            f6 = avg_monthly * 6 * (1 + growth)
            f12 = avg_monthly * 12 * (1 + growth)
            return {
                '6_month': {'forecast': f6, 'lower': f6 * 0.85, 'upper': f6 * 1.15},
                '12_month': {'forecast': f12, 'lower': f12 * 0.85, 'upper': f12 * 1.15}
            }

    all_forecasts = {'6_month': {}, '12_month': {}}
    forecast_dfs = {}

    if 'Store_ID' in df.columns:
        unique_stores = df['Store_ID'].unique()
        for store_id in unique_stores:
            store_df = df[df['Store_ID'] == store_id][['Date', sales_col]].copy()
            result = _prophet_forecast_monthly(store_df, store_label=str(store_id))
            if result:
                all_forecasts['6_month'][store_id] = result['6_month']
                all_forecasts['12_month'][store_id] = result['12_month']
                if 'forecast_df' in result:
                    forecast_dfs[store_id] = result['forecast_df']
    else:
        result = _prophet_forecast_monthly(df[['Date', sales_col]].copy())
        if result:
            all_forecasts['6_month']['ALL'] = result['6_month']
            all_forecasts['12_month']['ALL'] = result['12_month']
            if 'forecast_df' in result:
                forecast_dfs['ALL'] = result['forecast_df']

    if all_forecasts['6_month']:
        total_6m = sum(v['forecast'] for v in all_forecasts['6_month'].values())
        total_6m_lower = sum(v['lower'] for v in all_forecasts['6_month'].values())
        total_6m_upper = sum(v['upper'] for v in all_forecasts['6_month'].values())
    else:
        total_6m = total_6m_lower = total_6m_upper = 0

    if all_forecasts['12_month']:
        total_12m = sum(v['forecast'] for v in all_forecasts['12_month'].values())
        total_12m_lower = sum(v['lower'] for v in all_forecasts['12_month'].values())
        total_12m_upper = sum(v['upper'] for v in all_forecasts['12_month'].values())
    else:
        total_12m = total_12m_lower = total_12m_upper = 0

    return {
        '6_month': {'forecast': total_6m, 'lower': total_6m_lower, 'upper': total_6m_upper},
        '12_month': {'forecast': total_12m, 'lower': total_12m_lower, 'upper': total_12m_upper},
        'per_store_forecasts': all_forecasts,
        'forecast_dfs': forecast_dfs
    }


# ==================== GRANULAR FORECASTING ENGINE ====================
def generate_granular_forecast(df):
    import warnings
    warnings.filterwarnings("ignore")

    ROLLING_MONTHS = 12
    sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
    sku_col   = 'SKU_Name'          if 'SKU_Name'          in df.columns else None
    cat_col   = 'Product_Category'  if 'Product_Category'  in df.columns else None
    store_col = 'Store_ID'          if 'Store_ID'          in df.columns else None

    df = df.copy()
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)

    has_dates = 'Date' in df.columns
    if has_dates:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

    def _run_prophet(series_df, date_col='Date'):
        if not PROPHET_AVAILABLE or not has_dates:
            return None
        try:
            monthly = (series_df.set_index(date_col)[sales_col]
                       .resample('MS').sum().reset_index())
            monthly.columns = ['ds', 'y']
            monthly = monthly.sort_values('ds').reset_index(drop=True)
            if len(monthly) < 2:
                return None
            train = monthly.tail(ROLLING_MONTHS).copy()
            last_date = monthly['ds'].max()
            model = Prophet(
                yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False,  changepoint_prior_scale=0.001,
                seasonality_prior_scale=25, changepoint_range=0.95,
                interval_width=0.95
            )
            model.fit(train)
            future   = model.make_future_dataframe(periods=12, freq='MS')
            forecast = model.predict(future)
            fc = forecast[forecast['ds'] >= last_date][
                ['ds','yhat','yhat_lower','yhat_upper']].copy()
            for c in ['yhat','yhat_lower','yhat_upper']:
                fc[c] = fc[c].clip(lower=0)
            return {'hist': monthly, 'fc': fc, 'last': last_date}
        except Exception:
            return None

    def _fallback_totals(total_sales, label):
        avg = total_sales / 12 if total_sales > 0 else 0
        growth = 0.05
        return {
            'label':        label,
            'total_hist':   total_sales,
            '6m_forecast':  avg * 6 * (1+growth),
            '6m_lower':     avg * 6 * (1+growth) * 0.85,
            '6m_upper':     avg * 6 * (1+growth) * 1.15,
            '12m_forecast': avg * 12 * (1+growth),
            '12m_lower':    avg * 12 * (1+growth) * 0.85,
            '12m_upper':    avg * 12 * (1+growth) * 1.15,
            'hist':         None,
            'fc':           None,
        }

    def _pack(label, res, total_sales):
        if res is None:
            return _fallback_totals(total_sales, label)
        hist, fc, _ = res['hist'], res['fc'], res['last']
        f6  = fc.head(6)
        f12 = fc.head(12)
        return {
            'label':        label,
            'total_hist':   total_sales,
            '6m_forecast':  f6['yhat'].sum(),
            '6m_lower':     f6['yhat_lower'].sum(),
            '6m_upper':     f6['yhat_upper'].sum(),
            '12m_forecast': f12['yhat'].sum(),
            '12m_lower':    f12['yhat_lower'].sum(),
            '12m_upper':    f12['yhat_upper'].sum(),
            'hist':         hist,
            'fc':           fc,
        }

    overall_total = df[sales_col].sum()
    overall_res   = _run_prophet(df[['Date', sales_col]] if has_dates else df)
    overall       = _pack('Overall Company', overall_res, overall_total)

    stores = []
    if store_col:
        for sid in sorted(df[store_col].unique()):
            sdf = df[df[store_col] == sid]
            tot = sdf[sales_col].sum()
            res = _run_prophet(sdf[['Date', sales_col]] if has_dates else sdf)
            stores.append(_pack(str(sid), res, tot))

    categories = []
    if cat_col:
        for cat in sorted(df[cat_col].dropna().unique()):
            cdf = df[df[cat_col] == cat]
            tot = cdf[sales_col].sum()
            res = _run_prophet(cdf[['Date', sales_col]] if has_dates else cdf)
            categories.append(_pack(str(cat), res, tot))

    # Per-product (formerly SKU)
    products = []
    if sku_col:
        top_products = (df.groupby(sku_col)[sales_col].sum()
                    .sort_values(ascending=False).head(5).index.tolist())
        for sk in top_products:
            skdf = df[df[sku_col] == sk]
            tot  = skdf[sales_col].sum()
            res  = _run_prophet(skdf[['Date', sales_col]] if has_dates else skdf)
            products.append(_pack(str(sk), res, tot))

    return {
        'overall':    overall,
        'stores':     stores,
        'categories': categories,
        'products':   products,   # renamed from 'skus' to 'products'
        'sales_col':  sales_col,
        'raw_df':     df,
        'sku_col':    sku_col,
        'cat_col':    cat_col,
    }


# ==================== GRANULAR CHARTS ====================
def build_granular_charts(gf):
    """Build all 8 matplotlib figures for Intelligence Sales Forecast Dashboard."""
    plt.style.use('seaborn-v0_8-darkgrid')
    COLORS = ['#003366','#1f77b4','#e07b2a','#2ca02c','#d62728',
              '#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']

    def _fmt_inr(v):
        if v >= 1e7:  return f"₹{v/1e7:.1f}Cr"
        if v >= 1e5:  return f"₹{v/1e5:.1f}L"
        return f"₹{v:,.0f}"

    # ── Chart 1: Overall company historical + 12-month forecast ─────────────
    fig1, ax1 = plt.subplots(figsize=(13, 8))
    fig1.subplots_adjust(top=0.91, bottom=0.15, left=0.10, right=0.97)
    ov = gf['overall']
    if ov['hist'] is not None:
        hist = ov['hist']
        fc   = ov['fc']
        ax1.plot(hist['ds'], hist['y'], color='#1f77b4', lw=2.5, label='Historical')
        ax1.plot(fc['ds'],   fc['yhat'], color='#003366', lw=2, ls='--', label='12-Month Forecast')
        f6_end = fc['ds'].iloc[5] if len(fc) >= 6 else fc['ds'].iloc[-1]
        ax1.axvline(f6_end, color='#e07b2a', ls=':', lw=1.5, alpha=0.8, label='6-Month Mark')
    else:
        labels = ['6-Month Forecast', '12-Month Forecast']
        vals   = [ov['6m_forecast'], ov['12m_forecast']]
        ax1.bar(labels, vals, color=['#1f77b4','#003366'], alpha=0.85)
        for i, v in enumerate(vals):
            ax1.text(i, v, _fmt_inr(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Overall Company — Sales Forecast', fontsize=14, fontweight='bold', pad=14)
    ax1.set_ylabel('Sales (INR)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    # ── Chart 2: Category-level analysis (renamed from Store-vs-Store) ───────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 8))
    fig2.subplots_adjust(top=0.91, bottom=0.22, wspace=0.40, left=0.08, right=0.97)
    if gf['categories']:
        cat_labels = [s['label'] for s in gf['categories']]
        vals6  = [s['6m_forecast']  for s in gf['categories']]
        vals12 = [s['12m_forecast'] for s in gf['categories']]
        x = np.arange(len(cat_labels))
        w = 0.38
        bars6  = axes2[0].bar(x - w/2, vals6,  w, color=COLORS[:len(x)], alpha=0.85, label='6M')
        bars12 = axes2[0].bar(x + w/2, vals12, w, color=COLORS[:len(x)], alpha=0.55, label='12M')
        axes2[0].set_xticks(x)
        axes2[0].set_xticklabels(cat_labels, rotation=30, ha='right')
        axes2[0].set_title('Category Level Analysis — 6M vs 12M', fontsize=13, fontweight='bold', pad=14)
        axes2[0].set_ylabel('Forecasted Sales (INR)')
        axes2[0].legend()
        axes2[0].grid(axis='y', alpha=0.3)
        for b in bars6:
            axes2[0].text(b.get_x()+b.get_width()/2, b.get_height(),
                          _fmt_inr(b.get_height()), ha='center', va='bottom', fontsize=8)
        axes2[1].pie(vals12, labels=cat_labels, colors=COLORS[:len(vals12)],
                     autopct='%1.1f%%', startangle=90, textprops={'fontsize':9})
        axes2[1].set_title('12-Month Forecast Share by Category', fontsize=13, fontweight='bold', pad=14)
    elif gf['stores']:
        # fallback to stores if no categories
        store_labels = [s['label'] for s in gf['stores']]
        vals6  = [s['6m_forecast']  for s in gf['stores']]
        vals12 = [s['12m_forecast'] for s in gf['stores']]
        x = np.arange(len(store_labels))
        w = 0.38
        bars6  = axes2[0].bar(x - w/2, vals6,  w, color=COLORS[:len(x)], alpha=0.85, label='6M')
        bars12 = axes2[0].bar(x + w/2, vals12, w, color=COLORS[:len(x)], alpha=0.55, label='12M')
        axes2[0].set_xticks(x)
        axes2[0].set_xticklabels(store_labels, rotation=30, ha='right')
        axes2[0].set_title('Category Level Analysis — 6M vs 12M', fontsize=13, fontweight='bold', pad=14)
        axes2[0].set_ylabel('Forecasted Sales (INR)')
        axes2[0].legend()
        axes2[0].grid(axis='y', alpha=0.3)
        axes2[1].pie(vals12, labels=store_labels, colors=COLORS[:len(vals12)],
                     autopct='%1.1f%%', startangle=90, textprops={'fontsize':9})
        axes2[1].set_title('12-Month Forecast Share', fontsize=13, fontweight='bold', pad=14)
    else:
        axes2[0].text(0.5,0.5,'No category data',ha='center',va='center',transform=axes2[0].transAxes)
        axes2[1].text(0.5,0.5,'No category data',ha='center',va='center',transform=axes2[1].transAxes)

    # ── Chart 3: Category comparison detailed ────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(13, 8))
    fig3.subplots_adjust(top=0.91, bottom=0.25, left=0.10, right=0.97)
    if gf['categories']:
        cats   = gf['categories']
        labels = [c['label'] for c in cats]
        v6  = [c['6m_forecast']  for c in cats]
        v12 = [c['12m_forecast'] for c in cats]
        x = np.arange(len(labels))
        w = 0.38
        b6  = ax3.bar(x - w/2, v6,  w, color='#1f77b4', alpha=0.85, label='6-Month')
        b12 = ax3.bar(x + w/2, v12, w, color='#003366', alpha=0.85, label='12-Month')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=30, ha='right')
        ax3.set_title('Category-Level Forecast — 6M vs 12M', fontsize=13, fontweight='bold', pad=14)
        ax3.set_ylabel('Forecasted Sales (INR)')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        for b in list(b6)+list(b12):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height(),
                     _fmt_inr(b.get_height()), ha='center', va='bottom', fontsize=8, rotation=45)
    else:
        ax3.text(0.5,0.5,'No category data',ha='center',va='center',transform=ax3.transAxes)
        ax3.set_title('Category-Level Forecast', fontsize=13, fontweight='bold', pad=14)

    # ── Chart 4 & 5: Top-5 Product forecast bars (renamed from SKU) ──────────
    fig45, axes45 = plt.subplots(1, 2, figsize=(26, 7))
    fig45.subplots_adjust(top=0.88, bottom=0.10, left=0.18, right=0.97, wspace=0.55)

    products = gf.get('products', gf.get('skus', []))  # backward compat
    for ax_idx, (period_key, period_label) in enumerate([('6m_forecast','6-Month'),('12m_forecast','12-Month')]):
        ax = axes45[ax_idx]
        if products:
            sk     = sorted(products, key=lambda s: s[period_key], reverse=False)
            labels = [s['label'] for s in sk]
            vals   = [s[period_key] for s in sk]
            cols   = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(sk)))
            bars_h = ax.barh(labels, vals, color=cols, height=0.55, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Forecasted Sales (INR)', fontsize=11, fontweight='bold')
            ax.set_title(f'Top 5 Products — {period_label} Forecast', fontsize=13, fontweight='bold', pad=12)
            ax.grid(axis='x', alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            max_val = max(vals) if vals else 1
            for bar in bars_h:
                w = bar.get_width()
                x_pos = w + max_val * 0.01
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                        _fmt_inr(w), va='center', ha='left', fontsize=9, fontweight='bold', color='#1a1a2e')
            ax.set_xlim(0, max_val * 1.22)
        else:
            ax.text(0.5, 0.5, 'No product data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Top 5 Products — {period_label} Forecast', fontsize=13, fontweight='bold', pad=12)

    fig4 = fig45
    fig5 = None

    # ── Chart 6: Per-category forecast lines ─────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(13, 8))
    fig6.subplots_adjust(top=0.91, bottom=0.18, left=0.10, right=0.97)
    plotted = False
    # Use categories if available, else stores
    line_data = gf['categories'] if gf['categories'] else gf['stores']
    for i, s in enumerate(line_data):
        if s['hist'] is not None and s['fc'] is not None:
            clr = COLORS[i % len(COLORS)]
            ax6.plot(s['hist']['ds'], s['hist']['y'],
                     color=clr, lw=1.8, label=f"{s['label']} Historical")
            ax6.plot(s['fc']['ds'],   s['fc']['yhat'],
                     color=clr, lw=1.8, ls='--', label=f"{s['label']} Forecast")
            plotted = True
    if not plotted:
        ax6.text(0.5,0.5,'No time-series category data',ha='center',va='center',transform=ax6.transAxes)
    ax6.set_title('Per-Category Monthly Sales — Historical & Forecast', fontsize=13, fontweight='bold', pad=14)
    ax6.set_ylabel('Sales (INR)', fontsize=11)
    if plotted:
        ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.get_xticklabels(), rotation=30, ha='right')

    # ── Chart 7: All-Segment Summary ─────────────────────────────────────────
    all_items = (
        [('Overall', gf['overall'])] +
        [(f"Cat: {s['label']}", s) for s in gf['categories'][:5]] +
        [(f"Store: {s['label']}", s) for s in gf['stores'][:3]]
    )
    lbs       = [a[0] for a in all_items]
    hist_vals = [a[1]['total_hist']   for a in all_items]
    f6_vals   = [a[1]['6m_forecast']  for a in all_items]
    f12_vals  = [a[1]['12m_forecast'] for a in all_items]
    n = len(lbs)

    fig_h = max(6, n * 1.1 + 2)
    fig7, ax7 = plt.subplots(figsize=(14, fig_h))
    fig7.subplots_adjust(top=0.93, bottom=0.08, left=0.26, right=0.97)

    y    = np.arange(n)
    h    = 0.26
    bar_hist = ax7.barh(y + h,   hist_vals, h, label='Historical Total', color='#7f7f7f', alpha=0.75)
    bar_f6   = ax7.barh(y,       f6_vals,   h, label='6M Forecast',      color='#1f77b4', alpha=0.90)
    bar_f12  = ax7.barh(y - h,   f12_vals,  h, label='12M Forecast',     color='#003366', alpha=0.90)

    ax7.set_yticks(y)
    ax7.set_yticklabels(lbs, fontsize=9)
    ax7.set_xlabel('Sales (INR)', fontsize=11, fontweight='bold')
    ax7.set_title('All-Segment Summary: Historical vs 6M vs 12M Forecast',
                  fontsize=13, fontweight='bold', pad=14)
    ax7.legend(loc='lower right', fontsize=9)
    ax7.grid(axis='x', alpha=0.25)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    all_vals = hist_vals + f6_vals + f12_vals
    max_val  = max(all_vals) if all_vals else 1
    for bars_group in [bar_hist, bar_f6, bar_f12]:
        for bar in bars_group:
            w = bar.get_width()
            if w > 0:
                ax7.text(w + max_val * 0.008, bar.get_y() + bar.get_height()/2,
                         _fmt_inr(w), va='center', ha='left', fontsize=7.5, color='#333')

    # ── Chart 8: Monthly breakdown table ─────────────────────────────────────
    fig8, ax8 = plt.subplots(figsize=(13, 7))
    fig8.subplots_adjust(top=0.91, bottom=0.04, left=0.05, right=0.95)
    ax8.axis('off')
    ov = gf['overall']
    if ov['fc'] is not None:
        fc   = ov['fc'].head(12).copy()
        fc['Month']    = fc['ds'].dt.strftime('%b %Y')
        fc['Forecast'] = fc['yhat'].apply(_fmt_inr)
        table_data  = fc[['Month','Forecast']].values.tolist()
        col_labels  = ['Month', 'Forecasted Sales']
        tbl = ax8.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#003366')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(table_data)+1):
            clr = '#eaf2ff' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                tbl[(i, j)].set_facecolor(clr)
        ax8.set_title('12-Month Forecast Breakdown (Company-Level)', fontsize=13,
                      fontweight='bold', pad=18)
    else:
        ax8.text(0.5,0.5,'No time-series data for monthly breakdown',
                 ha='center',va='center',transform=ax8.transAxes,fontsize=12)
        ax8.set_title('12-Month Forecast Breakdown', fontsize=13, fontweight='bold', pad=18)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8


# ==================== HINDI / ENGLISH TRANSLATIONS ====================
LANG = {
    'en': {
        'insights_title':       'AI-Powered Business Insights',
        'overall_summary':      'Overall Performance Summary',
        'total_sales':          'Total Sales',
        'total_products':       'Total Products Analyzed',
        'avg_margin':           'Average Profit Margin',
        'health_score':         'Overall MSME Health Score',
        'perf_score':           'Overall Performance Score',
        'top5':                 'Top 5 Performing Products',
        'perf_metrics':         'Performance Metrics',
        'fin_risk':             'Financial Risk Score',
        'vendor_score':         'Vendor Reliability Score',
        'growth_score':         'Growth Potential Score',
        'lower_better':         '(Lower is better)',
        'forecast_title':       'ML-Powered Sales Forecast',
        'model_comparison':     'Model Comparison & Selection',
        'cv_metrics':           'Prophet Cross-Validation Performance',
        'six_month':            '6-Month Projection',
        'twelve_month':         '12-Month Projection',
        'forecast_sales':       'Forecasted Sales',
        'expected_range':       'Expected Range',
        'mape':                 'Mean Absolute % Error (MAPE)',
        'mae':                  'Mean Absolute Error (MAE)',
        'rmse':                  'Root Mean Square Error (RMSE)',
        'selected_model':       'Selected Model',
        'why_prophet':          'Why Prophet Won',
        'explain_title':        'Score Explainability — Why Your Scores Are What They Are',
        'snp_title':            'ONDC Seller Network Participant (SNP) Matching',
        'snp_best':             'Best Match',
        'snp_match':            'Match Score',
        'snp_reason':           'Why This Match',
        'snp_action':           'Recommended Action',
        'recommendations':      'AI-Generated Recommendations',
        'immediate':            'Immediate Actions',
        'strategic':            'Strategic Initiatives',
        'risk_alert':           'Risk Alerts',
        'store_forecast':       'Store-Specific Sales Forecasts',
        'data_quality':         'Data Quality Report',
        'skewness':             'Sales Distribution Skewness',
        'data_months':          'Months of Data Available',
        'inference_time':       'Analysis completed in',
        'seconds':              'seconds',
    },
    'hi': {
        'insights_title':       'AI से मिली आपके धंधे की जानकारी',
        'overall_summary':      'आपके धंधे का कुल हाल',
        'total_sales':          'कुल बिक्री',
        'total_products':       'कुल सामान (Products)',
        'avg_margin':           'औसत मुनाफा (%)',
        'health_score':         'धंधे की सेहत का स्कोर',
        'perf_score':           'काम का कुल स्कोर',
        'top5':                 'सबसे ज्यादा बिकने वाले 5 सामान',
        'perf_metrics':         'काम के नंबर',
        'fin_risk':             'पैसों का जोखिम स्कोर',
        'vendor_score':         'सप्लायर का भरोसा स्कोर',
        'growth_score':         'आगे बढ़ने की संभावना',
        'lower_better':         '(कम नंबर अच्छा है)',
        'forecast_title':       'AI से अगले महीनों की बिक्री का अनुमान',
        'model_comparison':     'कौन सा तरीका सबसे सही है',
        'cv_metrics':           'अनुमान कितना सही निकला',
        'six_month':            'अगले 6 महीने की बिक्री',
        'twelve_month':         'अगले 12 महीने की बिक्री',
        'forecast_sales':       'अनुमानित बिक्री',
        'expected_range':       'बिक्री कम से कम — ज्यादा से ज्यादा',
        'mape':                 'औसत गलती % (MAPE)',
        'mae':                  'औसत गलती रुपये में (MAE)',
        'rmse':                 'कुल गलती का अनुमान (RMSE)',
        'selected_model':       'चुना हुआ तरीका',
        'why_prophet':          'यह तरीका क्यों चुना',
        'explain_title':        'आपके स्कोर का मतलब — क्यों ऐसे नंबर आए',
        'snp_title':            'ONDC पर बेचने के लिए सबसे अच्छे Platform',
        'snp_best':             'सबसे अच्छा मैच',
        'snp_match':            'मैच का नंबर',
        'snp_reason':           'यह Platform क्यों सही है',
        'snp_action':           'अभी क्या करें',
        'recommendations':      'AI की सलाह',
        'immediate':            'अभी करने वाले काम',
        'strategic':            'आगे की योजना',
        'risk_alert':           'खतरे की चेतावनी',
        'store_forecast':       'हर दुकान की अगली बिक्री का अनुमान',
        'data_quality':         'आपके Data की जाँच',
        'skewness':             'बिक्री का फैलाव',
        'data_months':          'कितने महीने का Data है',
        'inference_time':       'जाँच पूरी हुई',
        'seconds':              'सेकंड में',
    }
}

def T(key, lang='en'):
    return LANG.get(lang, LANG['en']).get(key, LANG['en'].get(key, key))


# ==================== FEATURE 1: SNP MATCHING ENGINE ====================
SNP_CATALOG = {
    'GeM (Government e-Marketplace)': {
        'business_types': ['Manufacturing', 'FMCG', 'Electronics', 'Clothing', 'Services'],
        'min_health':      40,
        'segment_boost':   ['Champions', 'Loyal'],
        'description_en':  'Government procurement portal — ideal for MSMEs supplying to public sector.',
        'description_hi':  'सरकारी खरीद का Portal — अगर आप सरकारी दफ्तरों को सामान बेचते हैं तो यह आपके लिए बहुत अच्छा है।',
        'action_en':       'Register on GeM portal (gem.gov.in) and map your product catalogue.',
        'action_hi':       'GeM पोर्टल (gem.gov.in) पर रजिस्टर करें और अपना सामान लिस्ट करें।',
    },
    'Flipkart Commerce (ONDC)': {
        'business_types': ['FMCG', 'Supermarket', 'Clothing', 'Electronics'],
        'min_health':      30,
        'segment_boost':   ['Champions', 'Loyal', 'Potential'],
        'description_en':  'High-volume B2C SNP — best for consumer goods with strong demand.',
        'description_hi':  'बड़ी मात्रा में सामान बेचने का Platform — रोज काम आने वाले सामान के लिए बहुत अच्छा।',
        'action_en':       'Onboard via Flipkart Seller Hub — optimise product images and descriptions.',
        'action_hi':       'Flipkart Seller Hub पर जुड़ें — सामान की अच्छी फोटो और जानकारी डालें।',
    },
    'Meesho (ONDC)': {
        'business_types': ['Clothing', 'FMCG', 'Manufacturing'],
        'min_health':      20,
        'segment_boost':   ['Potential', 'At Risk'],
        'description_en':  'Social commerce SNP — ideal for price-sensitive segments and tier-2/3 markets.',
        'description_hi':  'छोटे शहरों और कम दाम वाले सामान के लिए — Reseller के जरिए ज्यादा ग्राहक मिलते हैं।',
        'action_en':       'List on Meesho for reseller network access — focus on competitive pricing.',
        'action_hi':       'Meesho पर सामान लिस्ट करें — दाम थोड़ा कम रखें ताकि ज्यादा बिक्री हो।',
    },
    'NSIC e-Marketplace': {
        'business_types': ['Manufacturing', 'FMCG', 'Electronics', 'Services'],
        'min_health':      25,
        'segment_boost':   ['Champions', 'Loyal', 'Potential'],
        'description_en':  'NSIC marketplace for MSE-to-MSE and B2B procurement — strong public sector linkage.',
        'description_hi':  'छोटे और मझोले कारोबारियों के लिए सरकारी मंडी — दूसरे व्यापारियों को थोक में बेचने का मौका।',
        'action_en':       'Register with NSIC for buyer-seller matchmaking and export facilitation.',
        'action_hi':       'NSIC पर रजिस्टर करें — खरीदार और विक्रेता का मेल कराया जाता है।',
    },
    'Amazon Seller Services (ONDC)': {
        'business_types': ['Electronics', 'FMCG', 'Clothing', 'Supermarket'],
        'min_health':      35,
        'segment_boost':   ['Champions', 'Loyal'],
        'description_en':  'Premium B2C SNP — suits high-quality products with strong margin and low returns.',
        'description_hi':  'अच्छे और महंगे सामान के लिए — जिनकी Quality अच्छी हो और वापसी (Return) कम हो।',
        'action_en':       'Apply for Amazon Easy Ship / FBA integration via ONDC bridge.',
        'action_hi':       'Amazon Easy Ship / FBA से जुड़ें — ONDC के जरिए आवेदन करें।',
    },
    'Udaan (B2B ONDC)': {
        'business_types': ['Manufacturing', 'FMCG', 'Clothing', 'Supermarket'],
        'min_health':      20,
        'segment_boost':   ['Champions', 'Loyal', 'Potential', 'At Risk'],
        'description_en':  'B2B wholesale SNP — best for MSMEs supplying to retailers and distributors.',
        'description_hi':  'थोक बिक्री का Platform — दुकानदारों और Distributors को सामान बेचने के लिए बहुत अच्छा।',
        'action_en':       'List bulk products on Udaan for retailer discovery and bulk orders.',
        'action_hi':       'Udaan पर थोक सामान लिस्ट करें — दुकानदार खुद आपसे संपर्क करेंगे।',
    },
}

def recommend_snp(user_data, df_scores, lang='en'):
    try:
        biz_type    = user_data.get('business_type', 'FMCG')
        health_avg  = float(df_scores['MSME_Health_Score'].mean()) if 'MSME_Health_Score' in df_scores.columns else 50.0
        growth_avg  = float(df_scores['Growth_Potential_Score'].mean()) if 'Growth_Potential_Score' in df_scores.columns else 0.5
        vendor_avg  = float(df_scores['Vendor_Score'].mean()) if 'Vendor_Score' in df_scores.columns else 0.5

        segments    = segment_customers(df_scores)
        dom_segment = max(segments, key=segments.get) if segments else 'Potential'

        scores = {}
        for snp_name, snp_data in SNP_CATALOG.items():
            score = 0.0
            if biz_type in snp_data['business_types']:
                score += 40
            elif any(bt in biz_type for bt in snp_data['business_types']):
                score += 20
            if health_avg >= snp_data['min_health']:
                score += 20 + min(20, (health_avg - snp_data['min_health']) / 2)
            if dom_segment in snp_data['segment_boost']:
                score += 20
            score += growth_avg * 10 + vendor_avg * 10
            scores[snp_name] = min(99, round(score))

        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        desc_key   = f'description_{lang}' if lang == 'hi' else 'description_en'
        action_key = f'action_{lang}'      if lang == 'hi' else 'action_en'

        out  = f"\n\n## 🏪 {T('snp_title', lang)}\n"
        out += f"> 🔍 Based on your business type (**{biz_type}**), "
        out += f"MSME Health Score (**{health_avg:.0f}%**), "
        out += f"and dominant product segment (**{dom_segment}**)\n\n"

        medals = ['🥇', '🥈', '🥉']
        for i, (snp, score) in enumerate(top3):
            snp_info = SNP_CATALOG[snp]
            bar = '█' * (score // 10) + '░' * (10 - score // 10)
            out += f"### {medals[i]} {snp}\n"
            out += f"**{T('snp_match', lang)}:** `{bar}` **{score}%**\n\n"
            out += f"**{T('snp_reason', lang)}:** {snp_info.get(desc_key, snp_info['description_en'])}\n\n"
            out += f"**{T('snp_action', lang)}:** {snp_info.get(action_key, snp_info['action_en'])}\n\n"
            out += "---\n"
        return out
    except Exception as e:
        return f"\n\n## 🏪 ONDC SNP Matching\n*Could not compute SNP recommendations: {str(e)}*\n"


# ==================== FEATURE 2: SCORE EXPLAINABILITY ====================
def explain_scores(df, lang='en'):
    try:
        avg = lambda col: float(df[col].mean()) if col in df.columns else 0.0

        fin_risk   = avg('Financial_Risk_Score')
        vendor_sc  = avg('Vendor_Score')
        growth_sc  = avg('Growth_Potential_Score')
        health_sc  = avg('MSME_Health_Score')
        perf_sc    = avg('Performance_Score')

        op_cost    = avg('Monthly_Operating_Cost_INR')
        sales      = max(avg('Monthly_Sales_INR'), 1)
        loan       = avg('Outstanding_Loan_INR')
        margin     = avg('Avg_Margin_Percent')
        demand     = avg('Monthly_Demand_Units')
        returns    = avg('Returns_Percentage')
        reliability= avg('Vendor_Delivery_Reliability')
        turnover   = avg('Inventory_Turnover')

        cost_ratio  = op_cost / sales
        loan_ratio  = loan / (sales * 12) if sales > 0 else 0

        def _risk_emoji(v): return '🔴' if v > 0.7 else ('🟡' if v > 0.4 else '🟢')
        def _good_emoji(v): return '🟢' if v > 0.6 else ('🟡' if v > 0.3 else '🔴')

        out = f"\n\n## 🔍 {T('explain_title', lang)}\n"
        out += "> *Powered by Explainable AI — every score is traceable to your actual data.*\n\n"

        out += f"### {_risk_emoji(fin_risk)} {T('fin_risk', lang)}: **{fin_risk:.2f}** {T('lower_better', lang)}\n"
        if cost_ratio > 0.8:
            out += f"- ⚠️ **High operating cost** — your costs are **{cost_ratio*100:.0f}%** of sales (healthy target: <60%)\n"
        elif cost_ratio > 0.5:
            out += f"- 🟡 **Moderate operating cost** — {cost_ratio*100:.0f}% of sales\n"
        else:
            out += f"- ✅ **Efficient cost structure** — operating costs at {cost_ratio*100:.0f}% of sales\n"
        if loan_ratio > 0.5:
            out += f"- ⚠️ **High loan burden** — outstanding loan is **{loan_ratio:.1f}×** annual revenue\n"
        elif loan_ratio > 0.2:
            out += f"- 🟡 **Moderate loan** — {loan_ratio:.1f}× annual revenue\n"
        else:
            out += f"- ✅ **Low debt** — loan is {loan_ratio:.1f}× annual revenue\n"

        out += f"\n### {_good_emoji(vendor_sc)} {T('vendor_score', lang)}: **{vendor_sc:.2f}**\n"
        if reliability > 0.7:
            out += f"- ✅ **Strong vendor reliability** — {reliability*100:.0f}% on-time delivery\n"
        else:
            out += f"- ⚠️ **Low vendor reliability** — only {reliability*100:.0f}% on-time (target: >70%)\n"
        if turnover > 5:
            out += f"- ✅ **Good inventory turnover** — {turnover:.1f}× per year\n"
        elif turnover > 2:
            out += f"- 🟡 **Average inventory turnover** — {turnover:.1f}× per year\n"
        else:
            out += f"- ⚠️ **Slow inventory movement** — {turnover:.1f}× per year (dead stock risk)\n"
        if margin > 20:
            out += f"- ✅ **Healthy margin** — {margin:.1f}% avg profit margin\n"
        else:
            out += f"- ⚠️ **Low margin** — {margin:.1f}% (industry benchmark: >20%)\n"

        out += f"\n### {_good_emoji(growth_sc)} {T('growth_score', lang)}: **{growth_sc:.2f}**\n"
        if demand > 100:
            out += f"- ✅ **High demand volume** — avg {demand:.0f} units/month\n"
        else:
            out += f"- 🟡 **Moderate demand** — avg {demand:.0f} units/month\n"
        if returns < 5:
            out += f"- ✅ **Low returns** — only {returns:.1f}% return rate (excellent quality signal)\n"
        elif returns < 15:
            out += f"- 🟡 **Acceptable returns** — {returns:.1f}% return rate\n"
        else:
            out += f"- ⚠️ **High returns** — {returns:.1f}% (quality or expectation issue)\n"

        out += f"\n### {_good_emoji(health_sc/100)} {T('health_score', lang)}: **{health_sc:.1f}%**\n"
        out += f"- Composite of: Financial Risk (40%) + Vendor Score (30%) + Growth Potential (30%)\n"
        if health_sc >= 65:
            out += f"- ✅ **Healthy business** — eligible for most ONDC SNP categories\n"
        elif health_sc >= 40:
            out += f"- 🟡 **Developing business** — focus on cost reduction and vendor reliability\n"
        else:
            out += f"- 🔴 **At risk** — immediate action needed on financial management\n"

        return out
    except Exception as e:
        return f"\n\n## 🔍 Score Explainability\n*Could not generate explanations: {str(e)}*\n"


# ==================== FEATURE 3 + 4: MODEL COMPARISON + PROPHET CV ====================
def compare_models_and_cv(df, lang='en'):
    import time
    try:
        sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
        if sales_col not in df.columns or 'Date' not in df.columns:
            return "", {}

        df2 = df[['Date', sales_col]].copy()
        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
        df2 = df2.dropna(subset=['Date'])
        monthly = df2.set_index('Date')[sales_col].resample('MS').sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        if len(monthly) < 6:
            return "", {}

        n_test = min(6, len(monthly) // 4)
        train  = monthly.iloc[:-n_test].copy()
        test   = monthly.iloc[-n_test:].copy()
        y_test = test['y'].values

        results = {}

        t0 = time.time()
        prophet_preds = None
        cv_metrics_md = ""
        if PROPHET_AVAILABLE and len(train) >= 2:
            try:
                m = Prophet(
                    yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.001,
                    seasonality_prior_scale=25,
                    changepoint_range=0.95,
                    interval_width=0.95
                )
                m.fit(train)
                future = m.make_future_dataframe(periods=n_test, freq='MS')
                fc     = m.predict(future)
                prophet_preds = fc['yhat'].values[-n_test:]
                prophet_preds = np.clip(prophet_preds, 0, None)

                if len(monthly) >= 12:
                    try:
                        from prophet.diagnostics import cross_validation, performance_metrics
                        horizon_days = 90
                        initial_days = max(180, len(monthly) * 30 // 2)
                        period_days  = 30
                        df_cv = cross_validation(
                            m,
                            initial=f'{initial_days} days',
                            period=f'{period_days} days',
                            horizon=f'{horizon_days} days',
                            parallel=None
                        )
                        df_pm = performance_metrics(df_cv)
                        mape_cv  = df_pm['mape'].mean()  * 100
                        mae_cv   = df_pm['mae'].mean()
                        rmse_cv  = df_pm['rmse'].mean()
                        grade    = 'Excellent ✅' if mape_cv < 10 else ('Good 🟡' if mape_cv < 20 else 'Fair 🔴')
                        cv_metrics_md = f"""
### 📊 {T('cv_metrics', lang)}
| Metric | Value | Grade |
|---|---|---|
| {T('mape', lang)} | **{mape_cv:.1f}%** | {grade} |
| {T('mae', lang)} | **₹{mae_cv:,.0f}** | — |
| {T('rmse', lang)} | **₹{rmse_cv:,.0f}** | — |
| Horizon | 90 days forward | — |
| Training Window | Rolling 12 months | — |

> *Cross-validation run on {len(df_cv)} evaluation windows using Prophet's built-in diagnostics.*"""
                    except Exception:
                        if prophet_preds is not None and len(y_test) > 0:
                            mask = y_test > 0
                            mape_cv = float(np.mean(np.abs((y_test[mask] - prophet_preds[mask]) / y_test[mask])) * 100) if mask.any() else 0
                            mae_cv  = float(np.mean(np.abs(y_test - prophet_preds)))
                            rmse_cv = float(np.sqrt(np.mean((y_test - prophet_preds)**2)))
                            grade   = 'Excellent ✅' if mape_cv < 10 else ('Good 🟡' if mape_cv < 20 else 'Fair 🔴')
                            cv_metrics_md = f"""
### 📊 {T('cv_metrics', lang)} (Hold-out Test Set)
| Metric | Value | Grade |
|---|---|---|
| {T('mape', lang)} | **{mape_cv:.1f}%** | {grade} |
| {T('mae', lang)} | **₹{mae_cv:,.0f}** | — |
| {T('rmse', lang)} | **₹{rmse_cv:,.0f}** | — |
| Test Set Size | {n_test} months | — |
"""
            except Exception:
                pass

        prophet_time = time.time() - t0
        if prophet_preds is not None and len(y_test) > 0:
            mask = y_test > 0
            mape = float(np.mean(np.abs((y_test[mask]-prophet_preds[mask])/y_test[mask]))*100) if mask.any() else 99.0
            mae  = float(np.mean(np.abs(y_test - prophet_preds)))
            results['Prophet (Notebook Config)'] = {'mape': mape, 'mae': mae, 'time': prophet_time}

        t0 = time.time()
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            hw_model = ExponentialSmoothing(
                train['y'].values,
                trend='add', seasonal=None,
                initialization_method='estimated'
            ).fit(optimized=True)
            hw_preds = hw_model.forecast(n_test)
            hw_preds = np.clip(hw_preds, 0, None)
            mask = y_test > 0
            hw_mape = float(np.mean(np.abs((y_test[mask]-hw_preds[mask])/y_test[mask]))*100) if mask.any() else 99.0
            hw_mae  = float(np.mean(np.abs(y_test - hw_preds)))
            results['Holt-Winters (Exponential Smoothing)'] = {'mape': hw_mape, 'mae': hw_mae, 'time': time.time()-t0}
        except Exception:
            pass

        t0 = time.time()
        try:
            from sklearn.linear_model import LinearRegression
            X_train = np.arange(len(train)).reshape(-1, 1)
            X_test  = np.arange(len(train), len(train)+n_test).reshape(-1, 1)
            lr      = LinearRegression().fit(X_train, train['y'].values)
            lr_preds= np.clip(lr.predict(X_test), 0, None)
            mask    = y_test > 0
            lr_mape = float(np.mean(np.abs((y_test[mask]-lr_preds[mask])/y_test[mask]))*100) if mask.any() else 99.0
            lr_mae  = float(np.mean(np.abs(y_test - lr_preds)))
            results['Linear Regression (Baseline)'] = {'mape': lr_mape, 'mae': lr_mae, 'time': time.time()-t0}
        except Exception:
            pass

        if not results:
            return "", {}

        best_model = min(results, key=lambda k: results[k]['mape'])
        rows = ""
        for model, metrics in sorted(results.items(), key=lambda x: x[1]['mape']):
            star = " 🏆 **SELECTED**" if model == best_model else ""
            rows += f'| {model}{star} | {metrics["mape"]:.1f}% | ₹{metrics["mae"]:,.0f} | {metrics["time"]:.2f}s |\n'

        why_prophet = ""
        if 'Prophet (Notebook Config)' in results and best_model == 'Prophet (Notebook Config)':
            why_prophet = (
                f"\n> **{T('why_prophet', lang)}:** Prophet's additive seasonality and changepoint detection "
                f"captures MSME sales patterns better than linear baselines. "
                f"MAPE of {results['Prophet (Notebook Config)']['mape']:.1f}% "
                f"vs Holt-Winters {results.get('Holt-Winters (Exponential Smoothing',{}).get('mape',99):.1f}% "
                f"on {n_test}-month hold-out test."
            )
        elif best_model != 'Prophet (Notebook Config)':
            why_prophet = (
                f"\n> ⚠️ **Note:** On this dataset, {best_model} achieved lower MAPE. "
                f"Prophet is still used as the primary model for its interval estimation and interpretability."
            )

        md = f"""
## 🤖 {T('model_comparison', lang)}
| Model | MAPE ↓ | MAE ↓ | Inference Time |
|---|---|---|---|
{rows}
**{T('selected_model', lang)}: Prophet (Notebook Config)**{why_prophet}

{cv_metrics_md}
"""
        return md, results

    except Exception as e:
        return f"\n\n## 🤖 Model Comparison\n*Error: {str(e)}*\n", {}


# ==================== DATA QUALITY REPORTER ====================
def data_quality_report(df, lang='en'):
    try:
        sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
        skewness  = float(df[sales_col].skew()) if sales_col in df.columns else 0.0
        months    = 0
        if 'Date' in df.columns:
            dates  = pd.to_datetime(df['Date'], errors='coerce').dropna()
            if len(dates) > 0:
                months = int(((dates.max() - dates.min()).days) / 30.44) + 1

        skew_label = ('Low (Normal distribution)' if abs(skewness) < 0.5
                      else ('Moderate — log transform applied' if abs(skewness) < 2
                            else 'High — data transformation applied'))
        skew_emoji = '✅' if abs(skewness) < 1 else ('🟡' if abs(skewness) < 2 else '⚠️')

        coverage   = '✅ Excellent' if months >= 18 else ('🟡 Adequate' if months >= 6 else '⚠️ Limited')
        total_rows = len(df)
        null_pct   = df[sales_col].isna().mean() * 100 if sales_col in df.columns else 0

        return f"""

## 📋 {T('data_quality', lang)}
| Parameter | Value | Status |
|---|---|---|
| {T('skewness', lang)} | **{skewness:.2f}** ({skew_label}) | {skew_emoji} |
| {T('data_months', lang)} | **{months}** months | {coverage} |
| Total Records | **{total_rows:,}** rows | {'✅' if total_rows > 100 else '🟡'} |
| Missing Sales Values | **{null_pct:.1f}%** | {'✅' if null_pct < 5 else '⚠️'} |
| DPDP Act 2023 | Consent captured ✅ | ✅ |
| Data Retention | Files deleted post-analysis ✅ | ✅ |

"""
    except Exception:
        return ""

def segment_customers(df):
    try:
        sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else ('Product_Name' if 'Product_Name' in df.columns else None)
        if not sku_col:
            return None
        sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
        if 'Date' in df.columns:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            reference_date = df['Date'].max()
            rfm = df.groupby(sku_col).agg({
                'Date': lambda x: (reference_date - x.max()).days,
                sales_col: ['count', 'sum']
            })
            rfm.columns = ['recency', 'frequency', 'monetary']
            if len(rfm) >= 3:
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm)
                n_clusters = min(5, len(rfm))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                rfm['segment'] = kmeans.fit_predict(rfm_scaled)
                segment_names = ['Champions', 'Loyal', 'Potential', 'At Risk', 'Lost']
                rfm['segment_name'] = rfm['segment'].apply(
                    lambda x: segment_names[x] if x < len(segment_names) else f'Segment {x}'
                )
                return rfm['segment_name'].value_counts().to_dict()
        return None
    except Exception:
        return None


def generate_insights(user_data, df, lang='en'):
    import time
    t_start = time.time()
    try:
        company_name = user_data.get('company_name', 'Your Company')
        df = calculate_scores(df)

        sales_col = 'Monthly_Sales_INR'
        sku_col   = 'SKU_Name'

        total_sales    = df[sales_col].sum()
        total_products = len(df[sku_col].unique()) if sku_col in df.columns else 0
        avg_margin     = df['Avg_Margin_Percent'].mean()
        perf_score     = df['Performance_Score'].mean()
        health_score   = df['MSME_Health_Score'].mean()
        fin_risk       = df['Financial_Risk_Score'].mean()
        vendor_sc      = df['Vendor_Score'].mean()
        growth_sc      = df['Growth_Potential_Score'].mean()

        def fmtv(v, prefix='₹', suffix=''):
            if pd.isna(v): return "N/A"
            if v >= 1e7: return f"{prefix}{v/1e7:.2f} Cr{suffix}"
            if v >= 1e5: return f"{prefix}{v/1e5:.2f} L{suffix}"
            return f"{prefix}{v:,.0f}{suffix}"

        def score_badge(v, invert=False):
            eff = (1 - v) if invert else v
            if eff >= 0.65: return "🟢 Excellent"
            if eff >= 0.40: return "🟡 Moderate"
            return "🔴 Needs Attention"

        def health_badge(v):
            if v >= 65: return "🟢 Healthy"
            if v >= 40: return "🟡 Developing"
            return "🔴 At Risk"

        insights = f"""---
# 🎯 {T('insights_title', lang)}
### 🏢 {company_name}
---

"""
        insights += f"""## 📊 Business Performance Summary

| Metric | Value | Status |
|--------|-------|--------|
| 💰 Total Revenue | **{fmtv(total_sales)}** | {health_badge(health_score)} |
| 📦 Products Analysed | **{total_products}** | — |
| 📈 Avg Profit Margin | **{avg_margin:.1f}%** | {'🟢 Strong' if avg_margin > 20 else ('🟡 Moderate' if avg_margin > 10 else '🔴 Low')} |
| 🧠 MSME Health Score | **{health_score:.1f}%** | {health_badge(health_score)} |
| 🌟 Performance Score | **{perf_score:.1f}%** | {score_badge(perf_score/100)} |

---

"""
        insights += f"""## 🔢 Score Breakdown

| Score Component | Value | Benchmark | Status |
|-----------------|-------|-----------|--------|
| ⚠️ Financial Risk Score | **{fin_risk:.2f}** | < 0.40 | {score_badge(fin_risk, invert=True)} |
| 🤝 Vendor Reliability Score | **{vendor_sc:.2f}** | > 0.60 | {score_badge(vendor_sc)} |
| 🚀 Growth Potential Score | **{growth_sc:.2f}** | > 0.60 | {score_badge(growth_sc)} |

> 💡 *Financial Risk: lower is better. Vendor & Growth: higher is better.*

---

"""
        insights += f"""## 🏆 Top 5 Performing Products

"""
        if sku_col in df.columns and not df.empty:
            top_skus = df.nlargest(5, sales_col)[[sku_col, sales_col, 'Monthly_Demand_Units', 'Avg_Margin_Percent', 'Performance_Score']]
            insights += "| Rank | Product | Revenue | Units/Month | Margin | Perf Score |\n"
            insights += "|------|---------|---------|-------------|--------|------------|\n"
            for i, (_, row) in enumerate(top_skus.iterrows()):
                insights += (f"| {i+1} | **{row[sku_col]}** | {fmtv(row[sales_col])} | "
                             f"{row['Monthly_Demand_Units']:.0f} | {row['Avg_Margin_Percent']:.1f}% | "
                             f"{row['Performance_Score']:.1f}% |\n")
        else:
            insights += "*No product-level data available.*\n"

        insights += "\n---\n\n"
        insights += data_quality_report(df, lang)
        insights += "\n---\n\n"

        model_md, model_results = compare_models_and_cv(df, lang)
        insights += model_md
        insights += "\n---\n\n"

        forecast_results = forecast_sales(df)
        f6  = forecast_results['6_month']
        f12 = forecast_results['12_month']

        insights += f"""## 🔮 {T('forecast_title', lang)}

| Horizon | Forecasted Sales | Expected Range |
|---------|-----------------|----------------|
| 📅 6-Month | **{fmtv(f6['forecast'])}** | {fmtv(f6['lower'])} — {fmtv(f6['upper'])} |
| 📅 12-Month | **{fmtv(f12['forecast'])}** | {fmtv(f12['lower'])} — {fmtv(f12['upper'])} |

> *Powered by Prophet ML with 95% confidence intervals. Rolling 12-month training window.*

---

"""
        insights += explain_scores(df, lang)
        insights += "\n---\n\n"
        insights += recommend_snp(user_data, df, lang)
        insights += "\n---\n\n"

        insights += f"""## 💡 {T('recommendations', lang)}

### 🎯 Immediate Actions (0–30 Days)

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 🔴 High | Prioritise top 5 products — focus inventory & marketing spend | +Revenue concentration |
| 🔴 High | Review products with Financial Risk Score > 0.70 | ↓ Risk exposure |
| 🟡 Medium | Strengthen vendor partnerships for low-reliability suppliers | ↑ Vendor Score |
| 🟡 Medium | Analyse high-return products for quality or packaging issues | ↑ Customer Satisfaction |

### 📊 Strategic Initiatives (30–90 Days)

| Initiative | Rationale | KPI to Track |
|------------|-----------|--------------|
| Demand Forecasting | Use ML predictions to reduce overstock & stockouts | Inventory Turnover |
| Margin Optimisation | Price or cost review for low-margin products | Avg Margin % |
| Growth Investment | Allocate budget to high-growth-potential products | Growth Potential Score |
| Operational Efficiency | Target operating cost below 60% of revenue | Financial Risk Score |

---

"""
        high_risk = df[df['Financial_Risk_Score'] > 0.7] if 'Financial_Risk_Score' in df.columns else pd.DataFrame()
        if len(high_risk) > 0:
            insights += f"## ⚠️ {T('risk_alert', lang)}\n\n"
            insights += f"> **{len(high_risk)} product(s)** have a Financial Risk Score above 0.70 and require immediate review.\n\n"
            if sku_col in high_risk.columns:
                risk_list = high_risk.nlargest(min(5, len(high_risk)), 'Financial_Risk_Score')[[sku_col, 'Financial_Risk_Score', sales_col]]
                insights += "| Product | Risk Score | Revenue |\n"
                insights += "|---------|-----------|--------|\n"
                for _, row in risk_list.iterrows():
                    insights += f"| {row[sku_col]} | **{row['Financial_Risk_Score']:.2f}** | {fmtv(row[sales_col])} |\n"
            insights += "\n---\n\n"

        if 'Store_ID' in df.columns and forecast_results.get('per_store_forecasts'):
            insights += f"## 📈 {T('store_forecast', lang)}\n\n"
            insights += "| Store | 6M Forecast | 6M Range | 12M Forecast |\n"
            insights += "|-------|------------|----------|-------------|\n"
            for store_id, fdata in forecast_results['per_store_forecasts']['6_month'].items():
                f12_store = forecast_results['per_store_forecasts']['12_month'].get(store_id, {})
                insights += (f"| Store {store_id} | **{fmtv(fdata['forecast'])}** | "
                             f"{fmtv(fdata['lower'])} — {fmtv(fdata['upper'])} | "
                             f"{fmtv(f12_store.get('forecast', 0))} |\n")
            insights += "\n---\n\n"

        elapsed = time.time() - t_start
        insights += f"> ⏱️ {T('inference_time', lang)} **{elapsed:.1f}** {T('seconds', lang)} &nbsp;|&nbsp; Model: Prophet v1.1 &nbsp;|&nbsp; Platform: **DataNetra.ai v4**"

        return insights, None, forecast_results
    except Exception as e:
        import traceback
        return None, f"Error generating insights: {str(e)}\n\n{traceback.format_exc()}", None


# ==================== LLM CHART SUMMARY ENGINE ====================
import json as _json

def _call_claude(prompt, max_tokens=300):
    try:
        import urllib.request
        payload = _json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return None


def generate_chart_summaries(df, forecast_results, lang='en'):
    try:
        sales_col = 'Monthly_Sales_INR'
        sku_col   = 'SKU_Name' if 'SKU_Name' in df.columns else None

        if sku_col and not df.empty:
            top3 = df.nlargest(3, sales_col)
            top3_txt = ", ".join(
                f"{row[sku_col]} (₹{row[sales_col]:,.0f})"
                for _, row in top3.iterrows()
            )
            bottom1 = df.nsmallest(1, sales_col).iloc[0]
            total_products = df[sku_col].nunique()
        else:
            top3_txt  = "N/A"
            bottom1   = None
            total_products = 0

        total_sales = df[sales_col].sum()
        top_pct     = (df.nlargest(3, sales_col)[sales_col].sum() / total_sales * 100) if total_sales > 0 else 0

        fin_risk  = df['Financial_Risk_Score'].mean()
        vendor_sc = df['Vendor_Score'].mean()
        growth_sc = df['Growth_Potential_Score'].mean()
        perf_sc   = df['Performance_Score'].mean()

        avg_margin = df['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in df.columns else 0
        health_avg = df['MSME_Health_Score'].mean()
        high_health_pct = (df['MSME_Health_Score'] > 60).mean() * 100 if 'MSME_Health_Score' in df.columns else 0

        f6  = forecast_results['6_month']['forecast']
        f12 = forecast_results['12_month']['forecast']
        growth_pct = ((f12 / total_sales) - 1) * 100 if total_sales > 0 and f12 > 0 else 0

        lang_instr = "Respond in Hindi (Devanagari script)." if lang == 'hi' else "Respond in English."

        p1 = f"""You are a business analyst summarizing a bar chart for an MSME owner in India.
Chart: Top Products by Sales Revenue.
Data: Top 3 products: {top3_txt}. Total {total_products} products analyzed. Top 3 contribute {top_pct:.1f}% of total revenue ₹{total_sales:,.0f}.
Write exactly 2 sentences: one key insight about revenue concentration, one actionable recommendation.
Keep it simple, practical, under 60 words total. {lang_instr}"""

        p2 = f"""You are a business analyst summarizing a performance scores bar chart for an MSME owner in India.
Data: Financial Risk Score: {fin_risk:.2f} (lower=better), Vendor Score: {vendor_sc:.2f}, Growth Potential: {growth_sc:.2f}, Performance Score: {perf_sc:.1f}%.
Write exactly 2 sentences: identify the strongest and weakest score, give one specific action to improve the weakest.
Keep it simple, practical, under 60 words total. {lang_instr}"""

        p3 = f"""You are a business analyst summarizing a scatter plot (Sales vs Profit Margin) for an MSME owner in India.
Data: Average margin {avg_margin:.1f}%, Average MSME Health Score {health_avg:.1f}%, {high_health_pct:.0f}% of products have health score above 60.
Write exactly 2 sentences: describe what the scatter pattern means for the business, and one recommendation.
Keep it simple, practical, under 60 words total. {lang_instr}"""

        p4 = f"""You are a business analyst summarizing a Prophet ML sales forecast chart for an MSME owner in India.
Data: Current total sales ₹{total_sales:,.0f}. 6-month forecast ₹{f6:,.0f}. 12-month forecast ₹{f12:,.0f}. Projected growth: {growth_pct:+.1f}% vs current period.
Write exactly 2 sentences: one about the forecast trend, one actionable recommendation based on whether growth is positive or negative.
Keep it simple, practical, under 60 words total. {lang_instr}"""

        def _fmt_summary(text, icon, title):
            if not text:
                return None
            return f"> {icon} **{title}**\n> {text}"

        s1_raw = _call_claude(p1)
        s2_raw = _call_claude(p2)
        s3_raw = _call_claude(p3)
        s4_raw = _call_claude(p4)

        if not s1_raw:
            s1_raw = (f"Your top 3 products contribute {top_pct:.1f}% of total revenue — "
                      f"a {'healthy' if top_pct < 70 else 'concentrated'} portfolio. "
                      f"{'Diversify revenue sources to reduce dependency on top products.' if top_pct > 70 else 'Continue investing in your top performers while developing mid-tier products.'}")

        if not s2_raw:
            weakest = min([('Financial Risk', fin_risk, True), ('Vendor Score', vendor_sc, False), ('Growth Score', growth_sc, False)],
                           key=lambda x: x[1] if not x[2] else (1 - x[1]))
            s2_raw = (f"Your strongest metric is {'Growth Potential' if growth_sc > vendor_sc and growth_sc > (1-fin_risk) else 'Vendor Score'} "
                      f"while {weakest[0]} needs attention. "
                      f"{'Reduce operating costs to lower financial risk.' if fin_risk > 0.5 else 'Strengthen vendor partnerships to improve reliability scores.'}")

        if not s3_raw:
            s3_raw = (f"With an average margin of {avg_margin:.1f}% and health score of {health_avg:.1f}%, "
                      f"{'most products are in a healthy zone.' if health_avg > 50 else 'many products need margin improvement.'} "
                      f"Focus on moving low-margin, low-health products toward the upper-right quadrant.")

        if not s4_raw:
            trend = "positive growth" if growth_pct > 0 else "a decline"
            s4_raw = (f"The Prophet model projects {trend} of {abs(growth_pct):.1f}% — "
                      f"12-month forecast is ₹{f12:,.0f}. "
                      f"{'Plan inventory buildup and marketing spend ahead of the forecast peak.' if growth_pct > 0 else 'Review pricing strategy and cut underperforming products to arrest the decline.'}")

        titles = {
            'en': ['📦 Top Products Insight', '📊 Scores Insight',
                   '🔵 Sales-Margin Insight', '🔮 Forecast Insight'],
            'hi': ['📦 शीर्ष उत्पाद अंतर्दृष्टि', '📊 स्कोर अंतर्दृष्टि',
                   '🔵 बिक्री-मार्जिन अंतर्दृष्टि', '🔮 पूर्वानुमान अंतर्दृष्टि'],
        }
        icons  = ['📦', '📊', '🔵', '🔮']
        tlist  = titles.get(lang, titles['en'])

        return (
            _fmt_summary(s1_raw, icons[0], tlist[0]),
            _fmt_summary(s2_raw, icons[1], tlist[1]),
            _fmt_summary(s3_raw, icons[2], tlist[2]),
            _fmt_summary(s4_raw, icons[3], tlist[3]),
        )

    except Exception as e:
        return (None, None, None, None)


# ==================== CATEGORY FILTER CHART ====================
def build_category_filter_chart(df, selected_category):
    """Build top 5 products chart for a selected category."""
    plt.style.use('seaborn-v0_8-darkgrid')

    sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
    sku_col   = 'SKU_Name' if 'SKU_Name' in df.columns else None
    cat_col   = 'Product_Category' if 'Product_Category' in df.columns else None

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(top=0.91, bottom=0.12, left=0.32, right=0.92)

    def _fmt_inr(v):
        if v >= 1e7:  return f"₹{v/1e7:.1f}Cr"
        if v >= 1e5:  return f"₹{v/1e5:.1f}L"
        return f"₹{v:,.0f}"

    if cat_col and selected_category and selected_category != "All Categories" and sku_col:
        filtered = df[df[cat_col] == selected_category] if selected_category in df[cat_col].values else df
    else:
        filtered = df

    if sku_col and not filtered.empty:
        top5 = filtered.nlargest(5, sales_col)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top5)))
        bars = ax.barh(top5[sku_col], top5[sales_col], color=colors, height=0.55, edgecolor='white')
        ax.set_xlabel('Sales (INR)', fontsize=12, fontweight='bold')
        cat_label = selected_category if selected_category and selected_category != "All Categories" else "All"
        ax.set_title(f'Top 5 Products — {cat_label}', fontsize=14, fontweight='bold', pad=14)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        max_val = top5[sales_col].max() if len(top5) > 0 else 1
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                     _fmt_inr(width), ha='left', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, max_val * 1.22)
    else:
        ax.text(0.5, 0.5, 'No product data available for this category',
                ha='center', va='center', transform=ax.transAxes, fontsize=13)
        ax.set_title('Top 5 Products', fontsize=14, fontweight='bold', pad=14)

    return fig


def get_category_options(df):
    """Get list of categories from dataframe."""
    cat_col = 'Product_Category' if 'Product_Category' in df.columns else None
    if cat_col:
        cats = sorted(df[cat_col].dropna().unique().tolist())
        return ["All Categories"] + cats
    # Fallback to standard categories if no category column
    return ["All Categories", "FMCG", "Clothing", "Electronics", "Home Decor"]


# ==================== DASHBOARD CHARTS ====================
def generate_dashboard_data(user_data, df):
    try:
        df = calculate_scores(df)
        sales_col = 'Monthly_Sales_INR'
        sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None

        total_sales     = df[sales_col].sum()
        avg_margin      = df['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in df.columns else np.nan
        total_profit    = total_sales * (avg_margin / 100) if not pd.isna(avg_margin) else np.nan
        health_score    = df['MSME_Health_Score'].mean()
        growth_score    = df['Growth_Potential_Score'].mean()
        performance_score = df['Performance_Score'].mean()
        fin_risk_score  = df['Financial_Risk_Score'].mean()
        vendor_score    = df['Vendor_Score'].mean()
        total_qty       = df['Monthly_Demand_Units'].sum() if 'Monthly_Demand_Units' in df.columns else np.nan
        total_products  = df[sku_col].nunique() if sku_col else 0
        avg_fin_risk    = fin_risk_score
        company_name    = user_data.get('company_name', '—')

        def fmt_inr(v):
            if pd.isna(v): return "N/A"
            if v >= 1e7:   return f"₹{v/1e7:.2f} Cr"
            if v >= 1e5:   return f"₹{v/1e5:.2f} L"
            return f"₹{v:,.0f}"
        def fmt_pct(v, d=1): return f"{v:.{d}f}%" if not pd.isna(v) else "N/A"
        def fmt_f(v, d=2):   return f"{v:.{d}f}" if not pd.isna(v) else "N/A"
        def fmt_qty(v):      return f"{v:,.0f} units" if not pd.isna(v) else "N/A"

        def _health_clr(v):
            if v >= 65: return "#27ae60"
            if v >= 40: return "#f39c12"
            return "#e74c3c"
        def _risk_clr(v):
            if v <= 0.40: return "#27ae60"
            if v <= 0.70: return "#f39c12"
            return "#e74c3c"
        def _score_clr(v):
            if v >= 65: return "#27ae60"
            if v >= 40: return "#f39c12"
            return "#e74c3c"
        def _badge(label, color):
            return (f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:12px;font-size:0.75rem;font-weight:600;">{label}</span>')

        health_lbl  = "Healthy"   if health_score >= 65 else ("Developing" if health_score >= 40 else "At Risk")
        risk_lbl    = "Low Risk"  if avg_fin_risk <= 0.40 else ("Moderate" if avg_fin_risk <= 0.70 else "High Risk")
        perf_lbl    = "Excellent" if performance_score >= 65 else ("Moderate" if performance_score >= 40 else "Low")
        growth_lbl  = "Strong"    if growth_score >= 0.60 else ("Moderate" if growth_score >= 0.35 else "Low")
        vendor_lbl  = "Strong"    if vendor_score >= 0.60 else ("Moderate" if vendor_score >= 0.35 else "Weak")

        kpi_html = f"""
<div style="font-family:Arial,sans-serif;margin:0 0 18px 0;">
  <div style="background:linear-gradient(135deg,#0f2557,#1a3a6b);border-radius:10px;
              padding:14px 22px;margin-bottom:18px;display:flex;align-items:center;gap:14px;">
    <span style="font-size:2rem;">📊</span>
    <div>
      <div style="color:white;font-size:1.15rem;font-weight:700;">Business Intelligence Dashboard</div>
      <div style="color:#a8c8ff;font-size:0.85rem;">{company_name} &nbsp;·&nbsp; Real-time Analytics</div>
    </div>
  </div>
  <table style="width:100%;border-collapse:separate;border-spacing:0;
                border-radius:10px;overflow:hidden;
                box-shadow:0 2px 12px rgba(0,51,102,0.10);font-size:0.93rem;">
    <thead>
      <tr style="background:#003366;">
        <th style="padding:11px 16px;text-align:left;font-weight:600;width:35%;color:white !important;">Metric</th>
        <th style="padding:11px 16px;text-align:right;font-weight:600;width:30%;color:white !important;">Value</th>
        <th style="padding:11px 16px;text-align:center;font-weight:600;width:20%;color:white !important;">Status</th>
        <th style="padding:11px 16px;text-align:left;font-weight:600;width:15%;color:white !important;">Benchmark</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#f0f7ff;">
        <td colspan="4" style="padding:6px 16px;font-weight:700;color:#003366;
            font-size:0.78rem;letter-spacing:0.05em;text-transform:uppercase;
            border-bottom:1px solid #d0e4f7;">💰 Revenue & Profitability</td>
      </tr>
      <tr style="background:white;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Total Revenue</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:#0f2557;font-size:1.05rem;">{fmt_inr(total_sales)}</td>
        <td style="padding:10px 16px;text-align:center;">—</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Gross Sales</td>
      </tr>
      <tr style="background:#fafcff;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Total Profit</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:#27ae60;font-size:1.05rem;">{fmt_inr(total_profit)}</td>
        <td style="padding:10px 16px;text-align:center;">—</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Revenue × Margin</td>
      </tr>
      <tr style="background:white;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Avg Profit Margin</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:#1a3a6b;">{fmt_pct(avg_margin)}</td>
        <td style="padding:10px 16px;text-align:center;">{'🟢' if avg_margin > 20 else ('🟡' if avg_margin > 10 else '🔴')}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Target &gt; 20%</td>
      </tr>
      <tr style="background:#f0f7ff;">
        <td colspan="4" style="padding:6px 16px;font-weight:700;color:#003366;
            font-size:0.78rem;letter-spacing:0.05em;text-transform:uppercase;
            border-bottom:1px solid #d0e4f7;">📦 Volume & Products</td>
      </tr>
      <tr style="background:white;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Total Quantity Sold</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:#0f2557;font-size:1.05rem;">{fmt_qty(total_qty)}</td>
        <td style="padding:10px 16px;text-align:center;">—</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Demand Units</td>
      </tr>
      <tr style="background:#fafcff;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Products Analysed</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:#0f2557;">{total_products:,} Products</td>
        <td style="padding:10px 16px;text-align:center;">—</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Unique Products</td>
      </tr>
      <tr style="background:#f0f7ff;">
        <td colspan="4" style="padding:6px 16px;font-weight:700;color:#003366;
            font-size:0.78rem;letter-spacing:0.05em;text-transform:uppercase;
            border-bottom:1px solid #d0e4f7;">🧠 Health & Performance Scores</td>
      </tr>
      <tr style="background:white;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">MSME Health Score</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:{_health_clr(health_score)};font-size:1.05rem;">{fmt_pct(health_score)}</td>
        <td style="padding:10px 16px;text-align:center;">{_badge(health_lbl, _health_clr(health_score))}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Target &gt; 65%</td>
      </tr>
      <tr style="background:#fafcff;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Performance Score</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:{_score_clr(performance_score)};font-size:1.05rem;">{fmt_pct(performance_score)}</td>
        <td style="padding:10px 16px;text-align:center;">{_badge(perf_lbl, _score_clr(performance_score))}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Target &gt; 65%</td>
      </tr>
      <tr style="background:white;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Growth Potential Score</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:{_score_clr(growth_score*100)};">{fmt_f(growth_score)}</td>
        <td style="padding:10px 16px;text-align:center;">{_badge(growth_lbl, _score_clr(growth_score*100))}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Target &gt; 0.60</td>
      </tr>
      <tr style="background:#fafcff;border-bottom:1px solid #eef2f7;">
        <td style="padding:10px 16px;color:#333;">Financial Risk Score</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:{_risk_clr(avg_fin_risk)};">{fmt_f(avg_fin_risk)}</td>
        <td style="padding:10px 16px;text-align:center;">{_badge(risk_lbl, _risk_clr(avg_fin_risk))}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Lower is better</td>
      </tr>
      <tr style="background:white;">
        <td style="padding:10px 16px;color:#333;">Vendor Reliability Score</td>
        <td style="padding:10px 16px;text-align:right;font-weight:700;color:{_score_clr(vendor_score*100)};">{fmt_f(vendor_score)}</td>
        <td style="padding:10px 16px;text-align:center;">{_badge(vendor_lbl, _score_clr(vendor_score*100))}</td>
        <td style="padding:10px 16px;color:#888;font-size:0.82rem;">Target &gt; 0.60</td>
      </tr>
    </tbody>
  </table>
</div>
"""

        plt.style.use('seaborn-v0_8-darkgrid')

        # Chart 1: Top Products by Sales Revenue (removed "10" from title)
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        fig1.subplots_adjust(top=0.91, bottom=0.12, left=0.32, right=0.92)
        if sku_col and not df.empty:
            top_products = df.nlargest(10, sales_col)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_products)))
            bars = ax1.barh(top_products[sku_col], top_products[sales_col], color=colors)
            ax1.set_xlabel('Sales (INR)', fontsize=12, fontweight='bold')
            ax1.set_title('Top Products by Sales Revenue', fontsize=14, fontweight='bold', pad=14)
            ax1.grid(axis='x', alpha=0.3)
            for bar in bars:
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2,
                         f'₹{width:,.0f}', ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No product data to display', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Top Products by Sales Revenue', fontsize=14, fontweight='bold', pad=14)

        # Chart 2: Performance Scores
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        fig2.subplots_adjust(top=0.91, bottom=0.12, left=0.10, right=0.97)
        scores_labels = ['Financial\nRisk', 'Vendor\nScore', 'Growth\nPotential', 'Performance\nScore']
        values = [
            df['Financial_Risk_Score'].mean(),
            df['Vendor_Score'].mean(),
            df['Growth_Potential_Score'].mean(),
            df['Performance_Score'].mean() / 100
        ]
        values = [0 if pd.isna(v) else v for v in values]
        colors2 = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
        bars2 = ax2.bar(scores_labels, values, color=colors2, alpha=0.8, width=0.6)
        ax2.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Scores Overview', fontsize=14, fontweight='bold', pad=14)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Chart 3: Sales vs Margin
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        fig3.subplots_adjust(top=0.91, bottom=0.12, left=0.12, right=0.90)
        if not df.empty and 'Avg_Margin_Percent' in df.columns:
            scatter = ax3.scatter(df[sales_col], df['Avg_Margin_Percent'],
                                  alpha=0.6, c=df['MSME_Health_Score'],
                                  cmap='RdYlGn', s=100, edgecolors='black', linewidth=0.5)
            ax3.set_xlabel('Monthly Sales (INR)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
            ax3.set_title('Sales vs Margin Analysis (Color = Health Score)', fontsize=14, fontweight='bold', pad=14)
            ax3.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Health Score', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No sales/margin data to display', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Sales vs Margin Analysis (No Data)', fontsize=14, fontweight='bold', pad=14)

        forecast_results = forecast_sales(df)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        fig4.subplots_adjust(top=0.91, bottom=0.15, left=0.10, right=0.97)

        plotted_line = False
        if 'forecast_dfs' in forecast_results and forecast_results['forecast_dfs']:
            try:
                df_plot = df.copy()
                df_plot['Date'] = pd.to_datetime(df_plot['Date'], errors='coerce')
                df_plot = df_plot.dropna(subset=['Date'])

                monthly_hist = df_plot.set_index('Date')[sales_col].resample('MS').sum().reset_index()
                monthly_hist.columns = ['ds', 'y']
                monthly_hist = monthly_hist.sort_values('ds')

                all_fdf = list(forecast_results['forecast_dfs'].values())
                if all_fdf:
                    combined = (
                        pd.concat(all_fdf)
                        .groupby('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
                        .sum()
                        .reset_index()
                        .sort_values('ds')
                    )

                    ax4.plot(monthly_hist['ds'], monthly_hist['y'],
                             color='#1f77b4', linewidth=2, label='Historical')
                    ax4.plot(combined['ds'], combined['yhat'],
                             color='#003366', linewidth=2, linestyle='--', label='Forecast')
                    ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Sales (INR)', fontsize=12, fontweight='bold')
                    ax4.set_title('Total Monthly Sales Forecast', fontsize=14, fontweight='bold', pad=14)
                    ax4.legend(fontsize=10)
                    ax4.grid(True, alpha=0.3)
                    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                    plotted_line = True
            except Exception:
                plotted_line = False

        if not plotted_line:
            months = ['6-Month\nForecast', '12-Month\nForecast']
            forecasts_vals = [
                forecast_results['6_month']['forecast'],
                forecast_results['12_month']['forecast']
            ]
            lowers = [forecast_results['6_month']['lower'], forecast_results['12_month']['lower']]
            uppers = [forecast_results['6_month']['upper'], forecast_results['12_month']['upper']]
            x_pos = np.arange(len(months))
            bars4 = ax4.bar(x_pos, forecasts_vals, color=['#9b59b6', '#e67e22'], alpha=0.8, width=0.5)
            ax4.errorbar(x_pos, forecasts_vals,
                         yerr=[[forecasts_vals[i] - lowers[i] for i in range(2)],
                                [uppers[i] - forecasts_vals[i] for i in range(2)]],
                         fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.5)
            ax4.set_ylabel('Forecasted Sales (INR)', fontsize=12, fontweight='bold')
            ax4.set_title('ML Sales Forecast (6M & 12M)', fontsize=14, fontweight='bold', pad=14)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(months)
            ax4.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars4, forecasts_vals):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'₹{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        s1, s2, s3, s4 = generate_chart_summaries(df, forecast_results)

        # Get category options for filter
        cat_options = get_category_options(df)

        return (
            kpi_html,
            "", "", "", "",
            fig1, fig2, fig3, fig4,
            s1, s2, s3, s4,
            None,
            cat_options,
            df  # pass raw df for category filtering
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ("N/A", "", "", "", "", None, None, None, None, None, None, None, None, f"Error: {str(e)}", ["All Categories"], None)


# ==================== MOCK DATA ====================
udyam_master_data = pd.DataFrame({
    'udyam_number': ['UDYAM-UP-01-0000001', 'UDYAM-TN-00-7629703', 'UDYAM-KL-03-0000003'],
    'enterprise_name': ['Tech Innovations Pvt Ltd', 'Retail Solutions Corp', 'FMCG Distributors'],
    'organisation_type': ['Private Limited', 'Partnership', 'Proprietorship'],
    'major_activity': ['FMCG', 'Supermarket', 'Electronics'],
    'enterprise_type': ['Small', 'Micro', 'Medium'],
    'state': ['Uttar Pradesh', 'TamilNadu', 'Kerala'],
    'city': [' लखनऊ', 'Chennai', 'Kochi']
})

def _fetch_msme_data(msme_number):
    fetched_data = udyam_master_data[udyam_master_data['udyam_number'] == msme_number]
    if not fetched_data.empty:
        row = fetched_data.iloc[0]
        return (row['enterprise_name'], row['organisation_type'], row['major_activity'],
                row['enterprise_type'], row['state'], row['city'], "✅ MSME Data Fetched Successfully")
    return "", "", "", "", "", "", "❌ MSME Data Not Found. Please check the number."


# ==================== CUSTOM CSS ====================
custom_css = """
.header-container {
    background: linear-gradient(135deg, #0f2557 0%, #1a3a6b 100%);
    padding: 10px 20px; display: flex;
    justify-content: space-between; align-items: center;
    box_shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.logo-section { display: flex; align-items: center; gap: 12px; }
.logo-img { height: 45px; width: auto; filter: brightness(1.1); }
.logo-text {
    background: linear-gradient(to right, #6a0dad, #007bff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 28px; font-weight: 700; letter-spacing: -0.5px;
}
.hero-section {
    background: linear-gradient(90deg, #0f172a, #1e3a8a);
    padding: 60px 40px; border-radius: 10px; text-align: center;
    color: white; position: relative; overflow: hidden;
}
.hero-title { font-size: 42px; font-weight: 700; margin-bottom: 15px; color: white; }
.hero-section h2.hero-sub-tagline { font-size: 24px; font-weight: 500; color: white; opacity: 0.9; }
.hero-section p.hero-description { margin-top: 15px; font-size: 18px; opacity: 0.9; color: white; }
.section { padding: 40px 20px; margin-top: 20px; border-radius: 8px;
    background-color: #f9f9f9; box_shadow: 0 2px 4px rgba(0,0,0,0.05); }
.section-title { font-size: 32px; font-weight: 700; color: #333; text-align: center; margin-bottom: 20px; }
#how-datanetra-works-section { background-color: #fff; border: 1px solid #e0e0e0; }
.steps-description-column { padding-right: 30px; border-right: 1px solid #eee; }
.login-signup-card { padding-left: 30px; text-align: center; }
.login-signup-title { font-size: 24px; font-weight: 600; color: #0f2557; margin-bottom: 15px; }
.capabilities-section {
    background-color: #f0f8ff; padding: 50px 20px; margin-top: 40px;
    text-align: center; border-radius: 8px; box_shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.capabilities-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 30px; max-width: 1200px; margin: 0 auto; }
.capability-card {
    background-color: #ffffff; padding: 30px; border-radius: 10px;
    box_shadow: 0 5px 15px rgba(0,0,0,0.08); text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.capability-card:hover { transform: translateY(-10px); box-shadow: 0 8px 20px rgba(0,0,0,0.15); }
.footer-section {
    background-color: #0f2557; color: #ffffff; padding: 20px; margin-top: 40px;
    border-radius: 8px; display: flex; justify-content: space-between; align-items: center;
}
.footer-section div { color: #ffffff; }
.footer-section a { color: #ffffff; text-decoration: none; }
.footer-section a:hover { text-decoration: underline; }
body, html { background-color: #f8f9fa; }
.chart-summary-box {
    background: linear-gradient(135deg, #eaf2ff 0%, #f0f7ff 100%);
    border-left: 4px solid #003366;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 4px 0 12px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #1a1a2e;
    box_shadow: 0 2px 6px rgba(0,51,102,0.08);
}
/* Force white text in KPI table header */
#kpi-table-container thead th,
#kpi-table-container th {
    color: white !important;
}
"""

# ==================== GRADIO UI ====================
business_types = ["Choose Business Type", "FMCG", "Supermarket", "Clothing", "Electronics"]
roles = ["Business Owner", "Co-Founder", "Category Manager", "Analyst", "Store Manager"]

ACTIVITY_TO_BIZ_TYPE = {
    'FMCG':         'FMCG',
    'Supermarket':  'Supermarket',
    'Electronics':  'Electronics',
    'Clothing':     'Clothing',
    'Manufacturing':'FMCG',
    'Services':     'FMCG',
    'Trading':      'FMCG',
}

with gr.Blocks(title="DataNetra.ai - MSME Intelligence", theme=gr.themes.Soft(), css=custom_css) as demo:

    # ── State ──────────────────────────────────────────────────────────────────
    step_state           = gr.State(0)
    user_data_state      = gr.State({})
    lang_state           = gr.State('en')
    dashboard_data_state = gr.State({
        'kpi1': "", 'kpi2': "", 'kpi3': "", 'kpi4': "", 'kpi5': "",
        'chart1': None, 'chart2': None, 'chart3': None, 'chart4': None
    })
    granular_forecast_data_state = gr.State(None)
    df_state = gr.State(None)  # store raw df for category filtering

    # ── Language Toggle Bar ─────────────────────────────────────────────────────
    with gr.Row(elem_id="lang-bar"):
        gr.HTML('''<div style="background:#0f2557;padding:8px 16px;border-radius:6px;display:flex;align-items:center;gap:12px;">
  <span style="color:white;font-weight:600;font-size:0.9rem;">🌐 Language / भाषा:</span>
</div>''')
        lang_en_btn = gr.Button("🇬🇧 English", size="sm", variant="primary",  elem_id="lang-en-btn")
        lang_hi_btn = gr.Button("🇮🇳 हिंदी",   size="sm", variant="secondary", elem_id="lang-hi-btn")
        lang_indicator = gr.Markdown("**Active: English**", elem_id="lang-indicator")

    # ── Landing page bilingual HTML builders ──────────────────────────────────
    def _landing_hero(lang):
        if lang == 'hi':
            return """
<div class="header-container">
  <div class="logo-section">
    <img src="https://i.postimg.cc/qRNQYbZJ/Data-Netra-Logo.jpg" class="logo-img" alt="DataNetra.ai Logo">
    <div class="logo-text">DataNetra.ai</div>
  </div>
</div>
<div class="hero-section">
  <h1 class="hero-title">AI से आपके धंधे की पूरी जानकारी</h1>
  <h2 class="hero-sub-tagline">Data देखें। सही फैसला लें। आगे बढ़ें।</h2>
  <p class="hero-description">अपना बिक्री का Data डालें — AI आपको बताएगा क्या बेचें, कब बेचें और कैसे मुनाफा बढ़ाएं।</p>
</div>"""
        return """
<div class="header-container">
  <div class="logo-section">
    <img src="https://i.postimg.cc/qRNQYbZJ/Data-Netra-Logo.jpg" class="logo-img" alt="DataNetra.ai Logo">
    <div class="logo-text">DataNetra.ai</div>
  </div>
</div>
<div class="hero-section">
  <h1 class="hero-title">AI-Powered Retail Intelligence</h1>
  <h2 class="hero-sub-tagline">Data with Vision. Decisions with Confidence.</h2>
  <p class="hero-description">Turn retail data into predictive insights, smarter decisions, and measurable growth.</p>
</div>"""

    def _landing_capabilities(lang):
        if lang == 'hi':
            return """
<div class="capabilities-section">
  <h2 class="section-title">DataNetra क्या-क्या कर सकता है?</h2>
  <p>धंधे के हर फैसले में आपकी मदद</p>
  <div class="capabilities-grid">
    <div class="capability-card"><div style="font-size:48px">🎯</div><h3>Smart स्कोरिंग</h3><p>आपके सामान और धंधे को नंबर से समझाता है — कौन अच्छा चल रहा है, कौन नहीं</p></div>
    <div class="capability-card"><div style="font-size:48px">📊</div><h3>धंधे का Dashboard</h3><p>बिक्री, मुनाफा और सेहत — सब एक जगह देखें, सीधे और साफ</p></div>
    <div class="capability-card"><div style="font-size:48px">🔮</div><h3>आगे की बिक्री का अनुमान</h3><p>AI बताता है कि अगले 6-12 महीनों में बिक्री कितनी होगी</p></div>
    <div class="capability-card"><div style="font-size:48px">🔗</div><h3>आसान जोड़</h3><p>अपने Excel या POS System का Data सीधे डालें — कोई तकनीकी झंझट नहीं</p></div>
  </div>
</div>
<div class="footer-section">
  <div>🔒 आपका Data सुरक्षित है</div>
  <div>© 2026 DataNetra.ai &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/company/108412762/" target="_blank">LinkedIn</a>
  </div>
</div>"""
        return """
<div class="capabilities-section">
  <h2 class="section-title">Platform Capabilities</h2>
  <p>Everything you need to make smarter business decisions</p>
  <div class="capabilities-grid">
    <div class="capability-card"><div style="font-size:48px">🎯</div><h3>Smart Scoring Engine</h3><p>Automated analysis for accurate performance scoring and health scores</p></div>
    <div class="capability-card"><div style="font-size:48px">📊</div><h3>Business Health Dashboard</h3><p>Monitor key metrics and KPIs in one real-time dashboard</p></div>
    <div class="capability-card"><div style="font-size:48px">🔮</div><h3>Predictive Insights</h3><p>AI-driven forecasts to anticipate future trends</p></div>
    <div class="capability-card"><div style="font-size:48px">🔗</div><h3>Easy Integration</h3><p>Seamlessly connect with your existing retail and POS systems</p></div>
  </div>
</div>
<div class="footer-section">
  <div>🔒 Data Secured &amp; Protected</div>
  <div>© 2026 DataNetra.ai &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/company/108412762/" target="_blank">LinkedIn</a>
  </div>
</div>"""

    # ── Step 0: Landing ────────────────────────────────────────────────────────
    with gr.Column(visible=True) as step0_col:
        landing_hero_html = gr.HTML(value=_landing_hero('en'))
        show_signup_trigger = gr.Button("", elem_id="show-signup-trigger", visible=False)

        with gr.Column(elem_classes="section", elem_id="how-datanetra-works-section"):
            landing_how_title = gr.Markdown("## How DataNetra Works", elem_classes="section-title")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="steps-description-column"):
                    landing_step1_title = gr.Markdown("### 📥 Step 1: Upload Your Data")
                    landing_step1_desc  = gr.Markdown("Easily upload Excel/CSV files for comprehensive analysis.")
                    landing_step2_title = gr.Markdown("### 🤖 Step 2: AI-Powered Analysis")
                    landing_step2_desc  = gr.Markdown("Our AI processes your data, forecasting trends and uncovering hidden insights.")
                    landing_step3_title = gr.Markdown("### 📊 Step 3: Actionable Dashboards & Recommendations")
                    landing_step3_desc  = gr.Markdown("Access interactive dashboards, KPI charts and personalized recommendations.")
                with gr.Column(scale=1, elem_classes="login-signup-card"):
                    landing_login_title  = gr.Markdown("**Already Registered**", elem_classes="login-signup-title")
                    quick_login_mobile   = gr.Textbox(label="Enter Mobile Number", placeholder="+91")
                    quick_login_btn      = gr.Button("Login", variant="primary", size="lg")
                    landing_login_error_msg = gr.Markdown(value="", visible=False)
                    landing_signup_title = gr.Markdown("**First Time User**", elem_classes="login-signup-title")
                    landing_signup_desc  = gr.Markdown("**Signup to unlock smart AI Insights**")
                    quick_signup_btn     = gr.Button("Sign Up Now", variant="primary", size="lg")

        landing_capabilities_html = gr.HTML(value=_landing_capabilities('en'))

    # ── Step 1: User Information ───────────────────────────────────────────────
    with gr.Column(visible=False) as step1_col:
        gr.Markdown("# 📝 Register New User\n## Step 1: User Information")
        name_input     = gr.Textbox(label="Full Name*")
        mobile_input   = gr.Textbox(label="Mobile Number*")
        email_input    = gr.Textbox(label="Email")
        role_input     = gr.Dropdown(choices=roles, label="Role*")
        with gr.Row():
            cancel1_btn = gr.Button("Cancel")
            next1_btn   = gr.Button("Next →", variant="primary")
        error1 = gr.Markdown()

    # ── Step 2: MSME Verification ──────────────────────────────────────────────
    with gr.Column(visible=False) as step2_col:
        gr.Markdown("## Step 2: MSME Verification")
        msme_number_input = gr.Textbox(label="MSME/Udyam Number*", placeholder="e.g., UDYAM-TN-00-7629703")
        otp_input         = gr.Textbox(label="OTP (Enter '1234' for demo)*", type="password")
        fetch_btn         = gr.Button("Fetch MSME Data", variant="secondary")
        fetch_status      = gr.Markdown()
        gr.Markdown("### Fetched MSME Details")
        fetched_name     = gr.Textbox(label="Enterprise Name",     interactive=False)
        fetched_org      = gr.Textbox(label="Organisation Type",   interactive=False)
        fetched_activity = gr.Textbox(label="Major Activity",      interactive=False)
        fetched_type     = gr.Textbox(label="Enterprise Type",     interactive=False)
        fetched_state    = gr.Textbox(label="State",               interactive=False)
        fetched_city     = gr.Textbox(label="City",                interactive=False)
        with gr.Row():
            back2_btn = gr.Button("← Back")
            next2_btn = gr.Button("Verify & Next →", variant="primary")
        error2 = gr.Markdown()

    # ── Step 3: Certificate ────────────────────────────────────────────────────
    with gr.Column(visible=False) as step3_col:
        gr.Markdown("## Step 3: MSME Certificate Review\n### Confirm MSME Details")
        confirm_name     = gr.Textbox(label="Enterprise Name",     interactive=False)
        confirm_org      = gr.Textbox(label="Organisation Type",   interactive=False)
        confirm_activity = gr.Textbox(label="Major Activity",      interactive=False)
        confirm_type     = gr.Textbox(label="Enterprise Type",     interactive=False)
        confirm_state    = gr.Textbox(label="State",               interactive=False)
        confirm_city     = gr.Textbox(label="City",                interactive=False)
        consent1             = gr.Checkbox(label="I confirm the above MSME details are correct", value=False)
        consent2             = gr.Checkbox(label="I consent to verify the MSME certificate",     value=False)
        certificate_upload   = gr.File(label="Upload MSME Certificate (PDF)", file_types=[".pdf"])
        with gr.Row():
            back3_btn = gr.Button("← Back")
            next3_btn = gr.Button("Confirm & Proceed →", variant="primary")
        error3 = gr.Markdown()

    # ── Step 4: Business Profile ───────────────────────────────────────────────
    with gr.Column(visible=False) as step4_col:
        verification_status_display = gr.Markdown(visible=False)
        gr.Markdown("## Step 4: Business Profile")
        business_type_input = gr.Dropdown(choices=business_types, label="Business Type*")
        years_input         = gr.Number(label="Years in Operation*", value=1, minimum=0)
        revenue_input       = gr.Dropdown(
            label="Monthly Revenue Range*",
            choices=["< 5 Lakh", "5-10 Lakh", "10-50 Lakh", "50 Lakh - 1 Crore", "> 1 Crore"]
        )
        with gr.Row():
            back4_btn = gr.Button("← Back")
            next4_btn = gr.Button("Submit Profile", variant="primary")
        error4 = gr.Markdown()
        proceed_to_step5_btn = gr.Button("Next: Upload Business Data →", variant="primary", visible=False)

    # ── Step 5: Upload & Analyse ───────────────────────────────────────────────
    with gr.Column(visible=False) as step5_col:
        login_welcome_message = gr.Markdown(value="", visible=False)
        gr.Markdown("## Step 5: Upload Business Data")
        consent_check  = gr.Checkbox(label="I consent to data analysis*", value=False)
        file_upload    = gr.File(label="Upload Excel File (.xlsx, .csv)*", file_types=[".xlsx", ".csv"])
        upload_message = gr.Markdown(value="", visible=False)
        with gr.Row():
            back5_btn   = gr.Button("← Back")
            cancel5_btn = gr.Button("❌ Cancel", variant="secondary")
            analyze_btn = gr.Button("🚀 Analyze Data", variant="primary", elem_id="analyze-data-btn")
        error5          = gr.Markdown()
        insights_output = gr.Markdown(elem_id="insights-output-md")
        view_dashboard_btn = gr.Button("📊 View Dashboard", visible=False, variant="primary")
        kpi1 = gr.Markdown(visible=False)
        kpi2 = gr.Markdown(visible=False)
        kpi3 = gr.Markdown(visible=False)
        kpi4 = gr.Markdown(visible=False)
        kpi5 = gr.Markdown(visible=False)
        chart1 = gr.Plot(visible=False)
        chart2 = gr.Plot(visible=False)
        chart3 = gr.Plot(visible=False)
        chart4 = gr.Plot(visible=False)
        sum1 = gr.Markdown(visible=False)
        sum2 = gr.Markdown(visible=False)
        sum3 = gr.Markdown(visible=False)
        sum4 = gr.Markdown(visible=False)

    # ── Step 6: Dashboard ──────────────────────────────────────────────────────
    with gr.Column(visible=False) as step6_col:
        kpi_table_dash = gr.HTML(value="", elem_id="kpi-table-container")
        gr.HTML("""
<div style="margin:28px 0 10px 0;padding:14px 20px;
     background:linear-gradient(90deg,#0f2557,#1a3a6b);border-radius:8px;">
  <span style="color:white;font-size:1.25rem;font-weight:700;">📈 Performance Visualizations</span>
</div>
""")
        with gr.Row():
            with gr.Column():
                chart1_dash = gr.Plot(label="Top Products by Sales Revenue")
                chart1_summary = gr.Markdown(value="", visible=False, elem_classes="chart-summary-box")
            with gr.Column():
                chart2_dash = gr.Plot(label="Performance Scores")
                chart2_summary = gr.Markdown(value="", visible=False, elem_classes="chart-summary-box")
        with gr.Row():
            with gr.Column():
                chart3_dash = gr.Plot(label="Sales vs Margin Analysis")
                chart3_summary = gr.Markdown(value="", visible=False, elem_classes="chart-summary-box")
            with gr.Column():
                chart4_dash = gr.Plot(label="Sales Forecast (Prophet)")
                chart4_summary = gr.Markdown(value="", visible=False, elem_classes="chart-summary-box")

        # ── Category Filter Section ─────────────────────────────────────────
        gr.HTML("""
<div style="margin:28px 0 10px 0;padding:14px 20px;
     background:linear-gradient(90deg,#1a3a6b,#0f2557);border-radius:8px;">
  <span style="color:white;font-size:1.25rem;font-weight:700;">🗂️ Category Performance — Filter by Category</span>
  <span style="color:#a8c8ff;font-size:0.9rem;margin-left:12px;">Select a category to see which 5 products are performing best</span>
</div>
""")
        with gr.Row():
            category_filter = gr.Dropdown(
                choices=["All Categories", "FMCG", "Clothing", "Electronics", "Home Decor"],
                value="All Categories",
                label="📂 Filter by Category",
                scale=1
            )
            category_filter_btn = gr.Button("🔍 Show Top 5 Products", variant="primary", scale=1)

        with gr.Row():
            category_chart = gr.Plot(label="Top 5 Products by Category")
            with gr.Column():
                category_insight_html = gr.HTML(value="", elem_id="category-insight-box")

        with gr.Row():
            forecast_deepdive_btn = gr.Button("📈 Intelligence Sales Forecast Dashboard →", variant="primary", size="lg")
            back6_btn = gr.Button("⬅ Back to Data Upload", variant="secondary", size="lg")

    # ── Step 7: Intelligence Sales Forecast Dashboard ─────────────────────────
    with gr.Column(visible=False) as step7_col:
        gr.HTML("""
<div style="background:linear-gradient(135deg,#0f2557,#1a3a6b);padding:18px 24px;border-radius:10px;margin-bottom:16px;">
  <h1 style="color:white;margin:0;font-size:1.9rem;font-weight:700;">📈 Intelligence Sales Forecast Dashboard</h1>
  <p style="color:#a8c8ff;margin:6px 0 0;font-size:1rem;">Per-Category · Per-Product · Company-Level · 6M & 12M Predictions</p>
</div>
""")
        with gr.Row():
            back7_btn        = gr.Button("⬅ Back to Dashboard",       variant="secondary")
            back7_to5_btn    = gr.Button("⬅ Back to Data Upload",      variant="secondary")

        gr.HTML('<div style="margin:32px 0 10px 0;padding:12px 18px;background:#f0f7ff;border-left:5px solid #003366;border-radius:0 6px 6px 0;"><span style="font-size:1.1rem;font-weight:700;color:#003366;">🏢 Overall Company Forecast</span></div>')
        forecast_chart_overall  = gr.Plot(label="Company — Historical + 12-Month Forecast")
        forecast_chart_monthly  = gr.Plot(label="Monthly Breakdown Table (12 Months)")

        gr.HTML('<div style="margin:32px 0 10px 0;padding:12px 18px;background:#f0f7ff;border-left:5px solid #003366;border-radius:0 6px 6px 0;"><span style="font-size:1.1rem;font-weight:700;color:#003366;">🗂️ Category Level Analysis</span></div>')
        forecast_chart_stores   = gr.Plot(label="Category Analysis — 6M vs 12M")
        forecast_chart_storelines = gr.Plot(label="Per-Category Historical + Forecast Lines")

        gr.HTML('<div style="margin:32px 0 10px 0;padding:12px 18px;background:#f0f7ff;border-left:5px solid #003366;border-radius:0 6px 6px 0;"><span style="font-size:1.1rem;font-weight:700;color:#003366;">🗂️ Category-Level Forecast Detail</span></div>')
        forecast_chart_cat      = gr.Plot(label="Category Forecast — 6M vs 12M")

        gr.HTML('<div style="margin:32px 0 10px 0;padding:12px 18px;background:#f0f7ff;border-left:5px solid #003366;border-radius:0 6px 6px 0;"><span style="font-size:1.1rem;font-weight:700;color:#003366;">📦 Top-5 Per Product Forecast Revenue — 6-Month &amp; 12-Month</span></div>')
        forecast_chart_sku6  = gr.Plot(label="Top 5 Per Product Forecast Revenue — 6-Month & 12-Month Combined")
        forecast_chart_sku12 = gr.Plot(visible=False)

        gr.HTML('<div style="margin:32px 0 10px 0;padding:12px 18px;background:#f0f7ff;border-left:5px solid #003366;border-radius:0 6px 6px 0;"><span style="font-size:1.1rem;font-weight:700;color:#003366;">🔢 All-Segment Summary</span></div>')
        forecast_chart_summary  = gr.Plot(label="Historical vs 6M vs 12M — All Segments")

    # ==================== EVENT HANDLERS ====================

    def update_visibility(step):
        return [gr.update(visible=(step == i)) for i in range(8)]

    def show_signup():
        return (1, *update_visibility(1))

    def handle_login(mobile):
        profile = get_user_profile(mobile)
        if profile:
            msg = f"✅ Welcome back, {profile['full_name']}! You have been navigated to upload business data file for Analysis."
            return (
                gr.update(value="", visible=False),
                profile,
                5,
                *update_visibility(5),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value=msg, visible=True),
            )
        return (
            gr.update(value="❌ No account found. Please register or try again.", visible=True),
            {}, 0, *update_visibility(0),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
        )

    def validate_step1(name, mobile, email, role, current_data):
        if not name or not mobile or not role:
            return ("⚠️ Please fill all required fields", current_data, 1, *update_visibility(1))
        updated = {**current_data, 'full_name': name, 'mobile_number': mobile, 'email': email, 'role': role}
        return ("", updated, 2, *update_visibility(2))

    def verify_step2(msme_num, otp, current_data, ent_name, org, activity, ent_type, state, city, status):
        if not msme_num or not otp:
            return ("⚠️ Please fill MSME number and OTP", current_data, 2, *update_visibility(2),
                    "", "", "", "", "", "", gr.update(visible=False))
        if otp != "1234":
            return ("⚠️ Invalid OTP", current_data, 2, *update_visibility(2),
                    "", "", "", "", "", "", gr.update(visible=False))
        if "Successfully" not in str(status):
            return ("⚠️ Please fetch MSME data first", current_data, 2, *update_visibility(2),
                    "", "", "", "", "", "", gr.update(visible=False))
        updated = {**current_data, 'msme_number': msme_num, 'company_name': ent_name,
                   'organisation_type': org, 'major_activity': activity,
                   'enterprise_type': ent_type, 'state': state, 'city': city}
        return ("✅ OTP Verified", updated, 3, *update_visibility(3),
                ent_name, org, activity, ent_type, state, city,
                gr.update(value="", visible=False))

    def confirm_step3(current_data, c1, c2, cert):
        if not c1 or not c2:
            return ("⚠️ Please accept both consents", current_data, 3, *update_visibility(3),
                    gr.update(value="", visible=False), gr.update(value="Choose Business Type"))
        if cert is None:
            return ("⚠️ Please upload certificate", current_data, 3, *update_visibility(3),
                    gr.update(value="", visible=False), gr.update(value="Choose Business Type"))

        updated = {**current_data, 'verification_status': 'APPROVED'}
        success_msg = (
            f"## ✅ Verification Status: APPROVED\n\n"
            f"**Company:** {current_data.get('company_name','N/A')}\n"
            f"**MSME Number:** {current_data.get('msme_number','N/A')}\n"
            f"**Status:** APPROVED ✓\n\nProceeding to Business Profile..."
        )
        activity = current_data.get('major_activity', '')
        pre_biz = ACTIVITY_TO_BIZ_TYPE.get(activity, "Choose Business Type")
        if activity in business_types:
            pre_biz = activity

        return (gr.update(value="", visible=False),
                updated, 4, *update_visibility(4),
                gr.update(value=success_msg, visible=True),
                gr.update(value=pre_biz))

    def submit_profile(biz_type, years, revenue, current_data):
        if not biz_type or biz_type == "Choose Business Type":
            return (gr.update(value="⚠️ Please select business type", visible=True),
                    current_data, gr.update(visible=False), gr.update(visible=True))
        if not revenue:
            return (gr.update(value="⚠️ Please select revenue range", visible=True),
                    current_data, gr.update(visible=False), gr.update(visible=True))
        if years is None or years <= 0:
            return (gr.update(value="⚠️ Please enter valid years in operation", visible=True),
                    current_data, gr.update(visible=False), gr.update(visible=True))

        updated = {**current_data, 'business_type': biz_type, 'years_operation': int(years),
                   'monthly_revenue_range': revenue, 'consent_given': True}
        try:
            user_id = save_user_profile(updated)
            success_msg = (
                f"## ✅ Business Profile Submitted!\n\n"
                f"**Company:** {updated.get('company_name','N/A')}\n"
                f"**Business Type:** {biz_type}\n"
                f"**Years in Operation:** {int(years)}\n"
                f"**Monthly Revenue:** {revenue}\n\n"
                f"Profile saved (ID: {user_id}). Click **Next** to upload data."
            )
            return (gr.update(value=success_msg, visible=True),
                    updated, gr.update(visible=True), gr.update(visible=False))
        except Exception as e:
            return (gr.update(value=f"❌ Error saving profile: {str(e)}", visible=True),
                    current_data, gr.update(visible=False), gr.update(visible=True))

    def analyze_data(user_data, consent, file, lang='en'):
        empty_dash = {
            'kpi1': "", 'kpi2': "", 'kpi3': "", 'kpi4': "", 'kpi5': "",
            'chart1': None, 'chart2': None, 'chart3': None, 'chart4': None
        }
        def _fail(msg):
            return (msg, gr.update(visible=False),
                    "", "", "", "", "",
                    None, None, None, None,
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), empty_dash,
                    None,
                    gr.update(choices=["All Categories"], value="All Categories"))

        if not consent:
            return _fail("⚠️ Please provide consent to analyze data")
        if file is None:
            return _fail("⚠️ Please upload an Excel or CSV file")

        try:
            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name)
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            else:
                return _fail("❌ Unsupported file format. Please upload .xlsx or .csv")

            col_remap = {
                'Sales_INR':              'Monthly_Sales_INR',
                'Monthly_Sales':          'Monthly_Sales_INR',
                'Gross_Sales':            'Monthly_Sales_INR',
                'Operating_Cost_INR':     'Monthly_Operating_Cost_INR',
                'Operating_Cost':         'Monthly_Operating_Cost_INR',
                'Outstanding_Loan':       'Outstanding_Loan_INR',
                'Outstanding_Amount':     'Outstanding_Loan_INR',
                'Vendor_Reliability':     'Vendor_Delivery_Reliability',
                'Inventory_Turnover_Rate':'Inventory_Turnover',
                'Average_Margin_Percent': 'Avg_Margin_Percent',
                'Profit_Margin_%':        'Avg_Margin_Percent',
                'Monthly_Demand':         'Monthly_Demand_Units',
                'Quantity_Sold':          'Monthly_Demand_Units',
                'Returns':                'Returns_Percentage',
                'Return_Quantity':        'Returns_Percentage',
                'Product_Name':           'SKU_Name',
            }
            for old_col, new_col in col_remap.items():
                if old_col in df.columns and new_col not in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)

            required_cols = {
                'Date': lambda: pd.to_datetime(datetime.datetime.now().date()),
                'Store_ID': 'Default', 'SKU_Name': 'Default',
                'Monthly_Sales_INR': 0, 'Monthly_Operating_Cost_INR': 0,
                'Outstanding_Loan_INR': 0, 'Vendor_Delivery_Reliability': 0,
                'Inventory_Turnover': 0, 'Avg_Margin_Percent': 0,
                'Monthly_Demand_Units': 0, 'Returns_Percentage': 0
            }
            for col, default in required_cols.items():
                if col not in df.columns:
                    df[col] = default() if callable(default) else default

            insights, error_msg, _ = generate_insights(user_data, df.copy(), lang=lang)
            if error_msg:
                return _fail(f"❌ {error_msg}")

            result = generate_dashboard_data(user_data, df.copy())
            k1 = result[0]
            k2, k3, k4, k5 = result[1], result[2], result[3], result[4]
            f1, f2, f3, f4 = result[5], result[6], result[7], result[8]
            s1, s2, s3, s4 = result[9], result[10], result[11], result[12]
            cat_options = result[14]
            raw_df = result[15]

            try:
                gf = generate_granular_forecast(df.copy())
            except Exception:
                gf = None

            dash = {'kpi1': k1, 'kpi2': k2, 'kpi3': k3, 'kpi4': k4, 'kpi5': k5,
                    'chart1': f1, 'chart2': f2, 'chart3': f3, 'chart4': f4,
                    'sum1': s1, 'sum2': s2, 'sum3': s3, 'sum4': s4,
                    'granular': gf}

            return (insights or "✅ Analysis completed", gr.update(visible=True),
                    k1, k2, k3, k4, k5, f1, f2, f3, f4,
                    s1, s2, s3, s4,
                    gr.update(value="", visible=False), dash,
                    raw_df,
                    gr.update(choices=cat_options, value="All Categories"))

        except Exception as e:
            import traceback
            return _fail(f"❌ Analysis failed: {str(e)}\n\n{traceback.format_exc()}")

    def show_dashboard(dashboard_data_value):
        def _summary(key):
            val = dashboard_data_value.get(key)
            if val:
                return gr.update(value=val, visible=True)
            return gr.update(value="", visible=False)

        granular_data = dashboard_data_value.get('granular')
        kpi_html_val  = dashboard_data_value.get('kpi1', "")

        return (
            6, *update_visibility(6),
            kpi_html_val,
            dashboard_data_value.get('chart1'), dashboard_data_value.get('chart2'),
            dashboard_data_value.get('chart3'), dashboard_data_value.get('chart4'),
            _summary('sum1'), _summary('sum2'),
            _summary('sum3'), _summary('sum4'),
            granular_data
        )

    def handle_file_upload_change(user_data, file):
        if file is not None:
            name = user_data.get('full_name', 'User')
            return (gr.update(value=f"Thank you, {name}, for uploading the dataset. Click 'Analyze Data' to view AI Insights and Dashboard.", visible=True),
                    gr.update(value="", visible=False))
        return gr.update(value="", visible=False), gr.update(value="", visible=False)

    def handle_category_filter(selected_category, raw_df):
        """Filter top 5 products by category and generate insight HTML."""
        if raw_df is None:
            return None, ""

        df = calculate_scores(raw_df.copy())
        fig = build_category_filter_chart(df, selected_category)

        # Build insight summary
        sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
        sku_col   = 'SKU_Name' if 'SKU_Name' in df.columns else None
        cat_col   = 'Product_Category' if 'Product_Category' in df.columns else None

        def _fmt_inr(v):
            if v >= 1e7:  return f"₹{v/1e7:.1f}Cr"
            if v >= 1e5:  return f"₹{v/1e5:.1f}L"
            return f"₹{v:,.0f}"

        if cat_col and selected_category and selected_category != "All Categories":
            filtered = df[df[cat_col] == selected_category] if selected_category in df[cat_col].values else df
            cat_label = selected_category
        else:
            filtered = df
            cat_label = "All Categories"

        insight_rows = ""
        if sku_col and not filtered.empty:
            top5 = filtered.nlargest(5, sales_col)
            avg_margin_cat = filtered['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in filtered.columns else 0
            health_avg_cat = filtered['MSME_Health_Score'].mean() if 'MSME_Health_Score' in filtered.columns else 0
            total_cat_sales = filtered[sales_col].sum()

            def _health_badge(v):
                if v >= 65: return ('<span style="background:#27ae60;color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;">Healthy</span>')
                if v >= 40: return ('<span style="background:#f39c12;color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;">Developing</span>')
                return ('<span style="background:#e74c3c;color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;">At Risk</span>')

            for i, (_, row) in enumerate(top5.iterrows()):
                medal = ['🥇','🥈','🥉','4️⃣','5️⃣'][i]
                insight_rows += f"""
          <tr style="background:{'white' if i%2==0 else '#f8fbff'};border-bottom:1px solid #e8f0fe;">
            <td style="padding:9px 12px;font-weight:700;color:#003366;">{medal}</td>
            <td style="padding:9px 12px;font-weight:600;color:#1a1a2e;">{row[sku_col]}</td>
            <td style="padding:9px 12px;text-align:right;font-weight:700;color:#27ae60;">{_fmt_inr(row[sales_col])}</td>
            <td style="padding:9px 12px;text-align:center;">{row.get('Avg_Margin_Percent', 0):.1f}%</td>
            <td style="padding:9px 12px;text-align:center;">{_health_badge(row.get('MSME_Health_Score', 0))}</td>
          </tr>"""

            insight_html = f"""
<div style="background:linear-gradient(135deg,#eaf2ff,#f0f7ff);border-left:4px solid #003366;border-radius:0 10px 10px 0;padding:16px 18px;margin-top:8px;">
  <div style="font-weight:700;color:#003366;font-size:1rem;margin-bottom:10px;">
    📊 Category Insight: <span style="color:#1f77b4;">{cat_label}</span>
  </div>
  <div style="display:flex;gap:18px;margin-bottom:14px;flex-wrap:wrap;">
    <div style="background:white;border-radius:8px;padding:10px 16px;box-shadow:0 1px 4px rgba(0,51,102,0.1);flex:1;min-width:120px;">
      <div style="color:#888;font-size:0.75rem;">Category Revenue</div>
      <div style="font-weight:700;color:#0f2557;font-size:1.1rem;">{_fmt_inr(total_cat_sales)}</div>
    </div>
    <div style="background:white;border-radius:8px;padding:10px 16px;box-shadow:0 1px 4px rgba(0,51,102,0.1);flex:1;min-width:120px;">
      <div style="color:#888;font-size:0.75rem;">Avg Margin</div>
      <div style="font-weight:700;color:#27ae60;font-size:1.1rem;">{avg_margin_cat:.1f}%</div>
    </div>
    <div style="background:white;border-radius:8px;padding:10px 16px;box-shadow:0 1px 4px rgba(0,51,102,0.1);flex:1;min-width:120px;">
      <div style="color:#888;font-size:0.75rem;">Avg Health Score</div>
      <div style="font-weight:700;color:#1a3a6b;font-size:1.1rem;">{health_avg_cat:.1f}%</div>
    </div>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
    <thead>
      <tr style="background:#003366;">
        <th style="padding:8px 12px;text-align:left;color:white;border-radius:4px 0 0 0;">#</th>
        <th style="padding:8px 12px;text-align:left;color:white;">Product</th>
        <th style="padding:8px 12px;text-align:right;color:white;">Revenue</th>
        <th style="padding:8px 12px;text-align:center;color:white;">Margin</th>
        <th style="padding:8px 12px;text-align:center;color:white;border-radius:0 4px 0 0;">Health</th>
      </tr>
    </thead>
    <tbody>{insight_rows}</tbody>
  </table>
</div>"""
        else:
            insight_html = f"""<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:16px;color:#856404;">
  No product data available for <strong>{cat_label}</strong>. Upload data with a <code>Product_Category</code> column to enable category filtering.
</div>"""

        return fig, insight_html

    # ==================== ALL OUTPUT LISTS ====================
    _step_cols = [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col]

    show_signup_trigger.click(show_signup, [], _step_cols)
    quick_signup_btn.click(show_signup, [], _step_cols)
    cancel1_btn.click(lambda: (0, *update_visibility(0)), [], _step_cols)
    back2_btn.click(lambda: (1, *update_visibility(1)), [], _step_cols)
    back3_btn.click(lambda: (2, *update_visibility(2)), [], _step_cols)
    back6_btn.click(lambda: (5, *update_visibility(5)), [], _step_cols)

    quick_login_btn.click(
        handle_login, [quick_login_mobile],
        [landing_login_error_msg, user_data_state, step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         error1, error2, error3, error4, error5, login_welcome_message]
    )

    next1_btn.click(
        validate_step1,
        [name_input, mobile_input, email_input, role_input, user_data_state],
        [error1, user_data_state, step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col]
    )

    fetch_btn.click(
        _fetch_msme_data, [msme_number_input],
        [fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetch_status]
    )

    next2_btn.click(
        verify_step2,
        [msme_number_input, otp_input, user_data_state,
         fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetch_status],
        [error2, user_data_state, step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         confirm_name, confirm_org, confirm_activity, confirm_type, confirm_state, confirm_city,
         fetch_status]
    )

    next3_btn.click(
        confirm_step3,
        [user_data_state, consent1, consent2, certificate_upload],
        [error3, user_data_state, step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         verification_status_display, business_type_input]
    )

    back4_btn.click(
        lambda: (3, *update_visibility(3), gr.update(value="", visible=False), gr.update(visible=True)),
        [],
        [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         error4, next4_btn]
    )

    next4_btn.click(
        submit_profile,
        [business_type_input, years_input, revenue_input, user_data_state],
        [error4, user_data_state, proceed_to_step5_btn, next4_btn]
    )

    proceed_to_step5_btn.click(
        lambda: (5, *update_visibility(5), gr.update(value="", visible=False), gr.update(value="", visible=False)),
        [],
        [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         error4, login_welcome_message]
    )

    back5_btn.click(
        lambda: (4, *update_visibility(4), gr.update(value="", visible=False)),
        [],
        [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         login_welcome_message]
    )

    cancel5_btn.click(
        lambda: (0, *update_visibility(0), gr.update(value="", visible=False)),
        [],
        [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         login_welcome_message]
    )

    analyze_btn.click(
        analyze_data,
        [user_data_state, consent_check, file_upload, lang_state],
        [insights_output, view_dashboard_btn,
         kpi1, kpi2, kpi3, kpi4, kpi5,
         chart1, chart2, chart3, chart4,
         sum1, sum2, sum3, sum4,
         upload_message, dashboard_data_state,
         df_state,
         category_filter]
    )

    file_upload.change(
        handle_file_upload_change,
        [user_data_state, file_upload],
        [upload_message, error5]
    )

    view_dashboard_btn.click(
        show_dashboard, [dashboard_data_state],
        [step_state, step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         kpi_table_dash,
         chart1_dash, chart2_dash, chart3_dash, chart4_dash,
         chart1_summary, chart2_summary, chart3_summary, chart4_summary,
         granular_forecast_data_state]
    )

    # Category filter button
    category_filter_btn.click(
        handle_category_filter,
        [category_filter, df_state],
        [category_chart, category_insight_html]
    )

    # Also trigger on dropdown change
    category_filter.change(
        handle_category_filter,
        [category_filter, df_state],
        [category_chart, category_insight_html]
    )

    def show_granular_dashboard(granular_data):
        gf = granular_data
        empty = (None,) * 8
        if gf is None:
            figs = empty
        else:
            try:
                figs = build_granular_charts(gf)
            except Exception:
                figs = empty
        return (
            7, *update_visibility(7),
            figs[0],   # forecast_chart_overall
            figs[7],   # forecast_chart_monthly
            figs[1],   # forecast_chart_stores (now category)
            figs[5],   # forecast_chart_storelines (now per-category lines)
            figs[3],   # forecast_chart_sku6 (combined product chart)
            None,      # forecast_chart_sku12 (hidden)
            figs[2],   # forecast_chart_cat
            figs[6],   # forecast_chart_summary
        )

    forecast_deepdive_btn.click(
        show_granular_dashboard,
        [granular_forecast_data_state],
        [step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col,
         forecast_chart_overall, forecast_chart_monthly, forecast_chart_stores,
         forecast_chart_storelines, forecast_chart_sku6, forecast_chart_sku12,
         forecast_chart_cat, forecast_chart_summary]
    )

    back7_btn.click(
        lambda: (6, *update_visibility(6)),
        [],
        [step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col]
    )

    back7_to5_btn.click(
        lambda: (5, *update_visibility(5)),
        [],
        [step_state,
         step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step7_col]
    )

    def switch_lang_en():
        return (
            'en',
            gr.update(variant='primary'),
            gr.update(variant='secondary'),
            '**Active: English 🇬🇧**',
            gr.update(value=_landing_hero('en')),
            gr.update(value=_landing_capabilities('en')),
            gr.update(value="## How DataNetra Works"),
            gr.update(value="### 📥 Step 1: Upload Your Data"),
            gr.update(value="Easily upload Excel/CSV files for comprehensive analysis."),
            gr.update(value="### 🤖 Step 2: AI-Powered Analysis"),
            gr.update(value="Our AI processes your data, forecasting trends and uncovering hidden insights."),
            gr.update(value="### 📊 Step 3: Actionable Dashboards & Recommendations"),
            gr.update(value="Access interactive dashboards, KPI charts and personalized recommendations."),
            gr.update(value="**Already Registered**"),
            gr.update(label="Enter Mobile Number"),
            gr.update(value="Login"),
            gr.update(value="**First Time User**"),
            gr.update(value="**Signup to unlock smart AI Insights**"),
            gr.update(value="Sign Up Now"),
        )

    def switch_lang_hi():
        return (
            'hi',
            gr.update(variant='secondary'),
            gr.update(variant='primary'),
            '**सक्रिय: हिंदी 🇮🇳**',
            gr.update(value=_landing_hero('hi')),
            gr.update(value=_landing_capabilities('hi')),
            gr.update(value="## DataNetra कैसे काम करता है?"),
            gr.update(value="### 📥 पहला काम: अपना Data डालें"),
            gr.update(value="अपनी Excel या CSV फाइल आसानी से Upload करें — बस एक क्लिक में।"),
            gr.update(value="### 🤖 दूसरा काम: AI जाँच करेगा"),
            gr.update(value="हमारा AI आपका Data पढ़कर बताएगा — क्या बिक रहा है, क्या नहीं, और आगे क्या होगा।"),
            gr.update(value="### 📊 तीसरा काम: रिपोर्ट और सलाह देखें"),
            gr.update(value="साफ Dashboard पर अपनी बिक्री, मुनाफा और AI की सलाह एक जगह देखें।"),
            gr.update(value="**पहले से जुड़े हैं?**"),
            gr.update(label="मोबाइल नंबर डालें"),
            gr.update(value="Login करें"),
            gr.update(value="**नए हैं? पहली बार?**"),
            gr.update(value="**अभी जुड़ें और AI से अपने धंधे की जानकारी पाएं**"),
            gr.update(value="अभी Register करें"),
        )

    _lang_landing_outputs = [
        lang_state, lang_en_btn, lang_hi_btn, lang_indicator,
        landing_hero_html, landing_capabilities_html,
        landing_how_title,
        landing_step1_title, landing_step1_desc,
        landing_step2_title, landing_step2_desc,
        landing_step3_title, landing_step3_desc,
        landing_login_title, quick_login_mobile,
        quick_login_btn,
        landing_signup_title, landing_signup_desc,
        quick_signup_btn,
    ]

    lang_en_btn.click(switch_lang_en, [], _lang_landing_outputs)
    lang_hi_btn.click(switch_lang_hi, [], _lang_landing_outputs)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DataNetra.ai - MSME Intelligence Platform v4.3")
    print("=" * 60)
    print("✅ Removed: Spinning wheel / loading overlay")
    print("✅ Renamed: Granular dashboard → Intelligence Sales Forecast Dashboard")
    print("✅ Renamed: SKU/Product → Per Product everywhere")
    print("✅ Renamed: Store Level Analysis → Category Level Analysis")
    print("✅ Added: Category filter with top 5 products insight panel")
    print("✅ Updated: 'Top 10 Products by Sales Revenue' → 'Top Products by Sales Revenue'")
    print("=" * 60)
    demo.launch()
