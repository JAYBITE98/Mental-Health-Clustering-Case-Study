# MENTAL HEALTH IN TECH - UNSUPERVISED LEARNING ANALYSIS
# Final version - no interactive display, just file output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
import os

warnings.filterwarnings('ignore')

# Turn off interactive display
plt.switch_backend('Agg')

print("=" * 60)
print("MENTAL HEALTH IN TECH SURVEY - CLUSTERING ANALYSIS")
print("=" * 60)

# ============================================
# STEP 1: LOAD THE DATASET
# ============================================

print("\n[STEP 1] Loading dataset...")
data_folder = 'data'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')] if os.path.exists(data_folder) else []

if len(csv_files) > 0:
    file_path = os.path.join(data_folder, csv_files[0])
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
else:
    print("❌ CSV file not found")
    exit()

# ============================================
# STEP 2: CLEAN THE DATA
# ============================================

print("\n[STEP 2] Cleaning data...")

if 'What is your age?' in df.columns:
    initial_count = len(df)
    df = df[(df['What is your age?'] >= 18) & (df['What is your age?'] <= 100)]
    print(f"   Removed {initial_count - len(df)} rows with invalid ages")

key_columns = [
    'What is your age?',
    'Are you self-employed?',
    'How many employees does your company or organization have?',
    'Does your employer provide mental health benefits as part of healthcare coverage?',
    'Do you know the options for mental health care available under your employer-provided coverage?',
    'Has your employer ever formally discussed mental health?',
    'Does your employer offer resources to learn more about mental health?',
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
    'Would you feel comfortable discussing a mental health disorder with your coworkers?',
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
    'Do you have medical coverage which includes treatment of mental health issues?'
]

existing_columns = [col for col in key_columns if col in df.columns]
df_clean = df[existing_columns].copy()
print(f"   Selected {len(existing_columns)} key columns")

# ============================================
# STEP 3: FAST ENCODING
# ============================================

print("\n[STEP 3] Converting text to numbers...")


def quick_encode(df):
    df_encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_encoded[col] = df[col]
            continue

        unique_vals = df[col].dropna().unique()
        mappings = {}

        for val in unique_vals:
            if pd.isna(val):
                mappings[val] = np.nan
            else:
                val_str = str(val).lower().strip()
                if val_str in ['yes', 'y', 'true']:
                    mappings[val] = 1
                elif val_str in ['no', 'n', 'false']:
                    mappings[val] = 0
                elif val_str in ['maybe', 'sometimes', "i don't know", 'unsure']:
                    mappings[val] = 0.5
                else:
                    # Use hash for other values
                    mappings[val] = (hash(val_str) % 100) / 100

        df_encoded[col] = df[col].map(mappings)

    return df_encoded


df_encoded = quick_encode(df_clean)
print(f"   Encoded {len(df_encoded.columns)} columns")

# ============================================
# STEP 4: HANDLE MISSING VALUES
# ============================================

print("\n[STEP 4] Handling missing values...")
missing_before = df_encoded.isnull().sum().sum()
print(f"   Missing values before: {missing_before}")

imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
print(f"   Missing values after: {df_imputed.isnull().sum().sum()}")

# ============================================
# STEP 5: SCALE
# ============================================

print("\n[STEP 5] Scaling features...")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)
print(f"   Shape for clustering: {df_scaled.shape}")

# ============================================
# STEP 6: CLUSTERING
# ============================================

print("\n[STEP 6] Clustering with k=4...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

print("\n📊 Cluster sizes:")
for i in range(4):
    size = (clusters == i).sum()
    percentage = (size / len(clusters)) * 100
    print(f"   Cluster {i}: {size} employees ({percentage:.1f}%)")

# ============================================
# STEP 7: PCA AND SAVE VISUALIZATION
# ============================================

print("\n[STEP 7] Creating and saving visualization...")
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Create and save plot (no display)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=30)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Employee Clusters Based on Mental Health Survey')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150)
plt.close()
print("   ✅ Visualization saved as 'cluster_visualization.png'")

# ============================================
# STEP 8: CLUSTER CHARACTERISTICS
# ============================================

print("\n[STEP 8] Cluster characteristics:")

df_with_clusters = df_imputed.copy()
df_with_clusters['Cluster'] = clusters

# Rename columns for cleaner output
column_renames = {
    'What is your age?': 'Age',
    'Does your employer provide mental health benefits as part of healthcare coverage?': 'Has Benefits',
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'Fears Consequences',
    'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'Comfort w Coworkers',
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'Comfort w Supervisor'
}

df_with_clusters = df_with_clusters.rename(
    columns={k: v for k, v in column_renames.items() if k in df_with_clusters.columns})

for i in range(4):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
    print(f"\n🔹 CLUSTER {i} ({len(cluster_data)} employees):")

    if 'Age' in cluster_data.columns:
        print(f"   • Average age: {cluster_data['Age'].mean():.1f} years")

    if 'Has Benefits' in cluster_data.columns:
        benefit_score = cluster_data['Has Benefits'].mean()
        benefit_text = "High" if benefit_score > 0.6 else "Medium" if benefit_score > 0.3 else "Low"
        print(f"   • Access to mental health benefits: {benefit_text} ({benefit_score:.2f})")

    if 'Fears Consequences' in cluster_data.columns:
        fear_score = cluster_data['Fears Consequences'].mean()
        fear_text = "High" if fear_score > 0.6 else "Medium" if fear_score > 0.3 else "Low"
        print(f"   • Fear of negative consequences: {fear_text} ({fear_score:.2f})")

    if 'Comfort w Coworkers' in cluster_data.columns:
        comfort_score = cluster_data['Comfort w Coworkers'].mean()
        comfort_text = "High" if comfort_score > 0.6 else "Medium" if comfort_score > 0.3 else "Low"
        print(f"   • Comfort discussing with coworkers: {comfort_text} ({comfort_score:.2f})")

    if 'Comfort w Supervisor' in cluster_data.columns:
        comfort_score = cluster_data['Comfort w Supervisor'].mean()
        comfort_text = "High" if comfort_score > 0.6 else "Medium" if comfort_score > 0.3 else "Low"
        print(f"   • Comfort discussing with supervisors: {comfort_text} ({comfort_score:.2f})")

# ============================================
# STEP 9: SAVE RESULTS
# ============================================

df_with_clusters.to_csv('cluster_assignments.csv', index=False)
print("\n📁 Saved files:")
print("   • cluster_visualization.png - Visual representation of clusters")
print("   • cluster_assignments.csv - Which cluster each employee belongs to")

# ============================================
# STEP 10: HR RECOMMENDATIONS
# ============================================

print("\n" + "=" * 60)
print("HR RECOMMENDATIONS FOR EACH CLUSTER")
print("=" * 60)

# Analyze which cluster needs what
recommendations = {}

for i in range(4):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]

    # Determine cluster type based on characteristics
    fear_high = False
    benefits_low = False
    comfort_low = False

    if 'Fears Consequences' in cluster_data.columns:
        if cluster_data['Fears Consequences'].mean() > 0.6:
            fear_high = True

    if 'Has Benefits' in cluster_data.columns:
        if cluster_data['Has Benefits'].mean() < 0.3:
            benefits_low = True

    if 'Comfort w Supervisor' in cluster_data.columns:
        if cluster_data['Comfort w Supervisor'].mean() < 0.3:
            comfort_low = True

    print(f"\n🔹 CLUSTER {i} - {len(cluster_data)} employees")

    if fear_high and comfort_low:
        print("   🚨 HIGH RISK GROUP")
        print("   → Need: Anonymous reporting systems, manager training, zero-retaliation policy")
    elif benefits_low:
        print("   ⚠️ RESOURCE GAP GROUP")
        print("   → Need: Expanded mental health benefits, better communication about existing resources")
    elif comfort_low:
        print("   📢 LOW COMFORT GROUP")
        print("   → Need: Mental health awareness campaigns, peer support programs")
    else:
        print("   ✅ RELATIVELY SUPPORTED GROUP")
        print("   → Need: Peer mentoring opportunities, maintain existing programs")

print("\n" + "=" * 60)
print("OVERALL RECOMMENDATIONS FOR HR DEPARTMENT")
print("=" * 60)
print("""
1. IMMEDIATE ACTIONS (next 30 days):
   • Implement anonymous mental health survey annually
   • Create mental health resource webpage for all employees
   • Train all managers on mental health first aid

2. SHORT-TERM GOALS (3-6 months):
   • Establish Employee Resource Group for mental health
   • Launch awareness campaign reducing stigma
   • Ensure all employees know available benefits

3. LONG-TERM STRATEGY (6-12 months):
   • Integrate mental health metrics into regular HR reports
   • Create peer support network
   • Review and expand mental health benefits package
""")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)

# Show the file locations
print("\n📂 Files saved in: " + os.getcwd())