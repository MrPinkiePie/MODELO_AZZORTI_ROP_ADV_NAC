import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

def calcular_psi(ref, nuevo, num_bins=10):
    """
    Calcula el Population Stability Index para una variable individual.
    <0.1: No hay cambio, 0.1-0.2: Cambio leve, >0.2: Cambio significativo.
    """
    # Eliminar NaNs
    ref = ref[~pd.isna(ref)] if isinstance(ref, pd.Series) else ref[~np.isnan(ref)]
    nuevo = nuevo[~pd.isna(nuevo)] if isinstance(nuevo, pd.Series) else nuevo[~np.isnan(nuevo)]
    
    if len(ref) == 0 or len(nuevo) == 0:
        return np.nan
        
    es_numerica = np.issubdtype(ref.dtype, np.number) if hasattr(ref, 'dtype') else isinstance(ref.iloc[0], (int, float))
    
    if es_numerica and len(np.unique(ref)) > num_bins:
        # Variables continuas: cortes cuantiles basados en referencia
        try:
            bins = np.percentile(ref, np.linspace(0, 100, num_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            ref_percents = pd.cut(ref, bins=bins, duplicates='drop').value_counts(normalize=True).sort_index().values
            nuevo_percents = pd.cut(nuevo, bins=bins, duplicates='drop').value_counts(normalize=True).sort_index().values
        except Exception:
            return np.nan
    else:
        # Categoricas o continuas de baja variabilidad
        valores_unicos = np.unique(np.concatenate([ref, nuevo]))
        ref_counts = pd.Series(ref).value_counts(normalize=True)
        nuevo_counts = pd.Series(nuevo).value_counts(normalize=True)
        
        ref_percents = np.array([ref_counts.get(v, 0) for v in valores_unicos])
        nuevo_percents = np.array([nuevo_counts.get(v, 0) for v in valores_unicos])
        
    # Evitar division por cero o log de cero
    eps = 1e-4
    ref_percents = np.where(ref_percents == 0, eps, ref_percents)
    nuevo_percents = np.where(nuevo_percents == 0, eps, nuevo_percents)
    
    psi_values = (nuevo_percents - ref_percents) * np.log(nuevo_percents / ref_percents)
    return np.sum(psi_values)

def reporte_psi(df_ref, df_new, variables):
    """
    Genera un DataFrame con el PSI de todas las variables solicitadas.
    """
    resultados = []
    for var in variables:
        if var in df_ref.columns and var in df_new.columns:
            psi_val = calcular_psi(df_ref[var], df_new[var])
            estado = 'PELIGRO' if psi_val > 0.2 else ('ALERTA' if psi_val > 0.1 else 'OK')
            resultados.append({'Variable': var, 'PSI': round(psi_val, 4), 'Estado': estado})
            
    if len(resultados) == 0:
        return pd.DataFrame(columns=['Variable', 'PSI', 'Estado'])
        
    df_res = pd.DataFrame(resultados).sort_values('PSI', ascending=False)
    return df_res

def graficar_reporte_psi(df_psi, export_path=None):
    """
    Genera un grafico de barras coloreado por el nivel de peligro del PSI.
    """
    plt.figure(figsize=(10, max(6, len(df_psi) * 0.3)))
    
    colores = {'OK': 'green', 'ALERTA': 'orange', 'PELIGRO': 'red'}
    bar_colors = [colores.get(estado, 'gray') for estado in df_psi['Estado']]
    
    sns.barplot(data=df_psi, x='PSI', y='Variable', hue='Variable', palette=bar_colors, legend=False)
    plt.axvline(x=0.1, color='orange', linestyle='--', label='Alerta (0.1)')
    plt.axvline(x=0.2, color='red', linestyle='--', label='Peligro (0.2)')
    
    plt.title('Population Stability Index (PSI) por Variable', fontsize=14)
    plt.xlabel('PSI Value (Mayor a 0.2 indica Drift Severo)')
    plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path)
    return plt.gca()

def calcular_mmd(X_ref, X_new, gamma=None, n_perm=100):
    """
    Calcula el Maximum Mean Discrepancy (MMD) multivariado entre dos datasets
    usando un kernel RBF (Gaussiano). Incluye heuristica de la mediana para gamma 
    y Test de Permutaciones para calcular el p-value.
    """
    X_ref = np.array(X_ref)
    X_new = np.array(X_new)
    
    # Muestreo para no colapsar la RAM O(N^2)
    max_samples = 2000
    if len(X_ref) > max_samples:
        np.random.seed(42)
        X_ref = X_ref[np.random.choice(len(X_ref), max_samples, replace=False)]
    if len(X_new) > max_samples:
        np.random.seed(42)
        X_new = X_new[np.random.choice(len(X_new), max_samples, replace=False)]
        
    if len(X_ref) == 0 or len(X_new) == 0:
        return {'mmd_score': np.nan, 'p_value': np.nan}
        
    if gamma is None:
        # Heuristica de la mediana
        from sklearn.metrics.pairwise import pairwise_distances
        X_cat = np.vstack([X_ref, X_new])
        if len(X_cat) > 1000:
            np.random.seed(42)
            X_cat = X_cat[np.random.choice(len(X_cat), 1000, replace=False)]
        dists = pairwise_distances(X_cat, metric='euclidean')
        median_dist = np.median(dists)
        if median_dist == 0:
            gamma = 1.0 / max(1, X_ref.shape[1])
        else:
            gamma = 1.0 / (2.0 * (median_dist ** 2))
            
    XX = rbf_kernel(X_ref, X_ref, gamma=gamma)
    YY = rbf_kernel(X_new, X_new, gamma=gamma)
    XY = rbf_kernel(X_ref, X_new, gamma=gamma)
    
    k_xx = XX.mean()
    k_yy = YY.mean()
    k_xy = XY.mean()
    mmd_squared = k_xx + k_yy - 2 * k_xy
    mmd_score = np.sqrt(max(mmd_squared, 0))
    
    # Test de Permutaciones
    mmd_null = []
    if n_perm > 0:
        X_pooled = np.vstack([X_ref, X_new])
        n_ref = len(X_ref)
        n_total = len(X_pooled)
        # Matriz de kernel completa para agilizar permutaciones
        K_full = rbf_kernel(X_pooled, X_pooled, gamma=gamma)
        
        np.random.seed(42)
        for _ in range(n_perm):
            idx = np.random.permutation(n_total)
            idx_ref = idx[:n_ref]
            idx_new = idx[n_ref:]
            
            k_xx_p = K_full[np.ix_(idx_ref, idx_ref)].mean()
            k_yy_p = K_full[np.ix_(idx_new, idx_new)].mean()
            k_xy_p = K_full[np.ix_(idx_ref, idx_new)].mean()
            
            mmd_null.append(np.sqrt(max(k_xx_p + k_yy_p - 2 * k_xy_p, 0)))
            
        umbral_95 = np.percentile(mmd_null, 95)
        p_value = np.mean([m >= mmd_score for m in mmd_null])
    else:
        umbral_95 = np.nan
        p_value = np.nan
        
    return {
        'mmd_score': mmd_score,
        'k_xx_intra_train': k_xx,
        'k_yy_intra_test': k_yy,
        'k_xy_cruzado': k_xy,
        'p_value': p_value,
        'umbral_critico_95': umbral_95,
        'gamma_usado': gamma
    }

def graficar_drift_multivariado_pca(X_train_scaled, X_test_scaled, export_path=None):
    """
    Genera un grafico 2D usando PCA para visualizar la separacion 
    multivariada entre el Train y el Test (Data Drift).
    """
    from sklearn.decomposition import PCA
    
    max_samples = 2000
    if len(X_train_scaled) > max_samples:
        np.random.seed(42)
        X_train_scaled = X_train_scaled[np.random.choice(len(X_train_scaled), max_samples, replace=False)]
    if len(X_test_scaled) > max_samples:
        np.random.seed(42)
        X_test_scaled = X_test_scaled[np.random.choice(len(X_test_scaled), max_samples, replace=False)]
        
    pca = PCA(n_components=2)
    # Proyectamos el espacio basandonos en lo que era "normal" (Train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.3, label='Train (2022-2025)', color='royalblue', s=15)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.5, label='Test (Reciente)', color='crimson', s=15)
    
    plt.title('Visualizacion de Drift Multivariado (PCA)', fontsize=14)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path)
    return plt.gca()
