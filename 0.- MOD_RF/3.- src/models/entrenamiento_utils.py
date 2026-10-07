"""
Utilidades para preparar los datos de entrenamiento y afinar el modelo de
ROP_ADV_NAC: split train/test por corte de campana, metrica de negocio
(servicio/sobrante), armado de X/y, seleccion de variables con Boruta y
tuning con Optuna (mismo enfoque simple que MODELO_AZZORTI_ROP_NAC_NAC: RF
dentro de un Pipeline con CatBoostEncoder, evaluado con R2 sobre TEST -- sin
seleccion de variables por Mutual Information ni folds temporales, para no
sumar complejidad que el equipo no estaba usando). No hacen fit del modelo
final (eso vive en entrenar.py, que ademas todavia no llega a ese paso -- ver
TODO en entrenar.py).
"""
import numpy as np
import pandas as pd
import category_encoders as ce
import optuna
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


def dividir_train_test_por_corte(df, col_camp='CAMP', corte=202508):
    """
    Divide TRAIN/TEST por un corte fijo de campana (a diferencia de
    dividir_train_test de MODELO_AZZORTI_ROP_NAC_NAC, que corta por fraccion
    80/20 cronologica): todo lo >= corte queda como TEST.
    """
    df = df.copy()
    df[col_camp] = df[col_camp].astype(int)
    train = df[df[col_camp] < corte].copy()
    test = df[df[col_camp] >= corte].copy()
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    return train, test


def invert_y(y_pred, base_df, ratio_col, mode='log_ratio'):
    """
    Recupera la prediccion a escala FACTOR_RCP_DESTINO (la unidad real del negocio).
    - raw       : y_pred
    - log       : exp(y_pred)
    - ratio     : y_pred * base_df[ratio_col]
    - log_ratio : exp(y_pred) * base_df[ratio_col]
    """
    if mode == 'raw':
        return y_pred
    elif mode == 'log':
        return np.exp(y_pred)
    elif mode == 'ratio':
        return y_pred * base_df[ratio_col].astype(float).values
    elif mode == 'log_ratio':
        return np.exp(y_pred) * base_df[ratio_col].astype(float).values
    else:
        raise ValueError(f'Y_MODE desconocido: {mode}')


def rf_score_con_std(qs_df, df_test, ratio_col, y_mode='log_ratio', col_factor_real='FACTOR_RCP_DESTINO', col_camp='CAMP'):
    """
    Metrica de negocio: nivel de servicio (SERVICIO) menos 3x el sobrante
    (SOBRANTE), penalizando ademas la variabilidad entre campanas (std). Muy
    similar en espiritu a mis_funciones.svs_sbr_por_grupo (servicio/sobrante
    por grupo), pero agregada globalmente y con el termino de std agregado
    para no premiar un modelo que acierta en promedio pero es inestable
    campana a campana.
    """
    df_eval = df_test.reset_index(drop=True).copy()

    q50_raw = qs_df[0.5].values
    q_mod = invert_y(q50_raw, df_eval, ratio_col=ratio_col, mode=y_mode)

    df_eval['Q_MOD'] = np.clip(q_mod, 0.0, None)
    df_eval[col_factor_real] = np.clip(df_eval[col_factor_real].astype(float).values, 0.0, None)

    df_eval['SERVICIO'] = np.minimum(df_eval['Q_MOD'], df_eval[col_factor_real])
    df_eval['SOBRANTE'] = np.maximum(df_eval['Q_MOD'] - df_eval[col_factor_real], 0.0)

    df_eval[col_camp] = df_eval[col_camp].astype(int)
    resumen = (
        df_eval.groupby(col_camp)
        .agg(SERVICIO_CAMP=('SERVICIO', 'sum'), SOBRANTE_CAMP=('SOBRANTE', 'sum'),
             FACT_CAMP=(col_factor_real, 'sum'))
        .assign(
            NIVEL_SRV=lambda d: d['SERVICIO_CAMP'] / np.where(d['FACT_CAMP'] == 0, np.nan, d['FACT_CAMP']),
            NIVEL_SBR=lambda d: d['SOBRANTE_CAMP'] / np.where(d['FACT_CAMP'] == 0, np.nan, d['FACT_CAMP']),
        )
        .reset_index()
    )

    fact_tot = resumen['FACT_CAMP'].sum()
    if fact_tot == 0:
        return -1e9, np.nan, np.nan, np.nan, np.nan

    nivel_srv = resumen['SERVICIO_CAMP'].sum() / fact_tot
    nivel_sbr = resumen['SOBRANTE_CAMP'].sum() / fact_tot
    std_srv = resumen['NIVEL_SRV'].std(skipna=True)
    std_sbr = resumen['NIVEL_SBR'].std(skipna=True)

    score = nivel_srv - 3 * nivel_sbr - 0.5 * (std_srv + std_sbr)
    return score, nivel_srv, nivel_sbr, std_srv, std_sbr


def armar_x_y(df, features):
    """
    Separa un DataFrame en matriz de variables (X) y target (y) segun el dict
    features (con llaves 'all' y 'target'). Igual que en
    MODELO_AZZORTI_ROP_NAC_NAC.
    """
    X = df[features['all']]
    y = df[features['target']]
    return X, y


def seleccionar_features_boruta(X_train, y_train, features, max_iter=100, seed=42, n_bootstrap=10, max_features=11, umbral_estabilidad=0.85):
    """
    Selección de variables usando SHAP Feature Importance.
    (La versión anterior de Boruta está comentada abajo por requerimiento de trazabilidad).
    
    1. Entrena un Random Forest rápido con TODAS las variables.
    2. Usa TreeExplainer de SHAP para calcular el impacto marginal absoluto de cada variable.
    3. Retiene el top N ('max_features').
    """
    print(f"⏳ Seleccionando Top {max_features} variables usando SHAP Feature Importance...")
    
    from sklearn.ensemble import RandomForestRegressor
    import category_encoders as ce
    import shap
    
    # Encodear categóricas una sola vez para el RF rápido
    encoder = ce.CatBoostEncoder(cols=features['cat'], random_state=seed)
    X_train_num = encoder.fit_transform(X_train, y_train)
    
    # Entrenar un RF rápido (100 árboles es suficiente para que SHAP converja)
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=seed, n_jobs=-1)
    rf_base.fit(X_train_num, y_train)
    
    # Calcular SHAP values
    explainer = shap.TreeExplainer(rf_base)
    # Tomamos una muestra aleatoria si el dataset es inmenso, pero para RF suele ser rápido
    shap_values = explainer.shap_values(X_train_num)
    
    # Importancia global SHAP: Promedio del valor absoluto de los SHAP values para cada variable
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    resumen = pd.DataFrame({
        "feature": X_train_num.columns,
        "shap_importance": mean_abs_shap
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    
    print("\nRanking de Importancia SHAP (Top 15):")
    print(resumen.head(15).to_string(index=False))
    
    # Toma de decisión
    features_top = resumen.head(max_features)['feature'].tolist()
    
    features_nuevas = dict(features)
    features_nuevas['num'] = [f for f in features_nuevas['num'] if f in features_top]
    features_nuevas['cat'] = [f for f in features_nuevas['cat'] if f in features_top]
    features_nuevas['all'] = features_nuevas['num'] + features_nuevas['cat']
    
    print(f"\nFeatures finales seleccionadas para el modelo ({len(features_nuevas['all'])}):", features_nuevas['all'])
    return features_nuevas


# =========================================================================================
# VERSIÓN ANTERIOR (BORUTA BOOTSTRAP) COMENTADA POR TRAZABILIDAD
# =========================================================================================
# def seleccionar_features_boruta(X_train, y_train, features, max_iter=100, seed=42, n_bootstrap=10, max_features=11, umbral_estabilidad=0.85):
#     """
#     Corre BorutaPy sobre X_train/y_train utilizando Bootstrap (remuestreo).
#     Si hay más de `max_features` variables (ej. Modelo RF), devuelve estrictamente el Top 11.
#     Si hay menos o igual cantidad de variables que el máximo (ej. Modelo de Tallas),
#     devuelve solo las variables cuya estabilidad supere el `umbral_estabilidad` (ej. 0.85).
#     """
#     n_features_candidatas = len(features['all'])
#     if n_features_candidatas <= max_features:
#         print(f"⏳ Corriendo Boruta Bootstrap ({n_bootstrap} iter.) para variables <= {max_features}. Filtro: Estabilidad >= {umbral_estabilidad}")
#     else:
#         print(f"⏳ Corriendo Boruta Bootstrap ({n_bootstrap} iter.) para encontrar el Top {max_features}...")
#     
#     from sklearn.base import clone
#     import warnings
#     
#     rng = np.random.RandomState(seed)
#     resultados = []
#     
#     rf_base = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=seed, n_jobs=-1)
#     
#     for b in range(n_bootstrap):
#         # Muestra con reemplazo
#         idx = rng.choice(len(X_train), size=len(X_train), replace=True)
#         
#         X_b = X_train.iloc[idx].reset_index(drop=True)
#         y_b = y_train.iloc[idx].reset_index(drop=True)
#         
#         # Encodear categóricas para esta muestra específica
#         encoder = ce.CatBoostEncoder(cols=features['cat'], random_state=seed + b)
#         X_b_num = encoder.fit_transform(X_b, y_b)
#         
#         rf_boruta = clone(rf_base)
#         boruta = BorutaPy(rf_boruta, n_estimators='auto', max_iter=max_iter, verbose=0, random_state=seed + b)
#         
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", UserWarning)
#             boruta.fit(X_b_num.values, y_b.values)
#             
#         resultados.append({
#             "support": boruta.support_,
#             "weak": boruta.support_weak_
#         })
#         
#     support = np.vstack([r["support"] for r in resultados])
#     # Calculamos estabilidad principal y sumamos un poco de peso (0.5) a las veces que salió como tentativa
#     weak = np.vstack([r["weak"] for r in resultados])
#     estabilidad = support.mean(axis=0) + (weak.mean(axis=0) * 0.5)
#     
#     nombres = features['all']
#     resumen = pd.DataFrame({
#         "feature": nombres,
#         "estabilidad": estabilidad
#     }).sort_values("estabilidad", ascending=False).reset_index(drop=True)
#     
#     print("\nRanking de Estabilidad Boruta Bootstrap (Top 15):")
#     print(resumen.head(15).to_string(index=False))
#     
#     # Toma de decisión
#     if n_features_candidatas <= max_features:
#         features_top = resumen[resumen['estabilidad'] >= umbral_estabilidad]['feature'].tolist()
#         if len(features_top) == 0:
#             print("⚠️ CUIDADO: Ninguna variable superó el umbral. Se retendrá la variable top 1 para evitar que el modelo explote.")
#             features_top = [resumen.iloc[0]['feature']]
#     else:
#         features_top = resumen.head(max_features)['feature'].tolist()
# 
#     
#     features = dict(features)
#     features['num'] = [f for f in features['num'] if f in features_top]
#     features['cat'] = [f for f in features['cat'] if f in features_top]
#     features['all'] = features['num'] + features['cat']
#     
#     print(f"\nFeatures finales seleccionadas para el modelo ({len(features['all'])}):", features['all'])
#     return features

def construir_objetivo_optuna(X_train, y_train, X_test, y_test, features, seed=42, train_camp=None):
    """
    Arma la funcion objetivo de Optuna: en cada trial define un RandomForest
    con hiperparametros sugeridos, lo entrena sobre train (dentro de un
    Pipeline con CatBoostEncoder) y devuelve el R2 obtenido sobre test. Mismo
    enfoque que MODELO_AZZORTI_ROP_NAC_NAC (sin seleccion de variables por
    Mutual Information dentro del trial).
    """
    def objetivo(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 300, 1200),
            'max_depth'        : trial.suggest_int('max_depth', 5, 25),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'max_features'     : trial.suggest_float('max_features', 0.1, 1.0),
        }

        pipe_optuna = Pipeline(steps=[
            ('encoder', ce.CatBoostEncoder(cols=features['cat'], random_state=seed)),
            ('modelo',  RandomForestRegressor(**params, bootstrap=True, random_state=seed))
        ])

        if train_camp is not None:
            # Ponderación Exponencial por Antigüedad
            decay_rate = trial.suggest_float('decay_rate', 0.0, 1.0)
            
            # Asumimos que train_camp.max() es la campaña más reciente en train (ej. 202511)
            # Para evitar que el gap numérico entre años afecte (ej 202501 - 202418), 
            # lo ideal sería usar un índice ordinal o simplificar con resta directa.
            # Convertimos a string, sacamos año y campaña para hacer una distancia real en campañas.
            camp_str = train_camp.astype(str)
            anios = camp_str.str[:4].astype(int)
            ciclos = camp_str.str[4:].astype(int)
            # Convertir a un índice de tiempo absoluto (cada año tiene 18 campañas)
            tiempo_absoluto = anios * 18 + ciclos
            distancia = tiempo_absoluto.max() - tiempo_absoluto
            
            sample_weights = np.exp(-decay_rate * distancia)
            pipe_optuna.fit(X_train, y_train, modelo__sample_weight=sample_weights)
        else:
            pipe_optuna.fit(X_train, y_train)
            
        y_pred = pipe_optuna.predict(X_test)

        return r2_score(y_test, y_pred)

    return objetivo


def ejecutar_estudio_optuna(objetivo, n_trials=100, seed=42):
    """
    Corre un estudio de Optuna (maximizando) sobre la funcion objetivo dada y
    devuelve los mejores hiperparametros encontrados. Copiada de
    MODELO_AZZORTI_ROP_NAC_NAC.

    n_trials: cantidad de combinaciones de hiperparametros que prueba Optuna
    (bajarlo reduce el tiempo de corrida, a costa de una busqueda menos fina).
    """
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objetivo, n_trials=n_trials)
    print(study.best_params)
    return study.best_params
