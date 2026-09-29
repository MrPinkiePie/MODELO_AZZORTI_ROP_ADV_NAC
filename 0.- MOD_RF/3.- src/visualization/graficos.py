"""
Funciones de graficos reutilizables entre notebooks de ROP_ADV_NAC, para no
repetir el mismo codigo de seaborn/matplotlib en cada notebook. Migradas
desde "0.- MOD_0.py" (EDA de atipicos/exposicion) y "analisis_ratio_precio.py"
(diagnostico de RATIO_PRECIO). Cada funcion recibe un DataFrame y devuelve el
objeto figura/eje (o el FacetGrid), para que quien la llame decida si
mostrarlo o exportarlo a 5.- reports/figures/.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def graficar_svs_sbr_por_bin(resultado_largo, orden=None):
    """
    Grafica un barplot de SVS/SBR (%) por bin de ratio (ej. Q70_Q30), a partir
    del formato largo (RATIO, METRICA, VALOR) que arma
    evaluacion.analizar_svs_sbr_por_bin_ratio. Devuelve el eje (ax). Copiada
    de MODELO_AZZORTI_ROP_NAC_NAC.
    """
    plt.figure(figsize=(10, 9))
    ax = sns.barplot(
        data=resultado_largo,
        x='RATIO',
        y='VALOR',
        hue='METRICA',
        order=orden,
    )

    # recorre cada grupo de barras y agrega la etiqueta
    for contenedor in ax.containers:
        ax.bar_label(contenedor, fmt='%.1f', padding=3)

    return ax


def graficar_distribucion_ratio(df, col_ratio, limite_superior=None, titulo=None):
    """Histograma de col_ratio (ej. RATIO_FACTOR), con linea opcional de umbral de atipicos."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col_ratio], bins=30, kde=True, stat='probability', ax=ax)
    if limite_superior is not None:
        ax.axvline(limite_superior, color='red', linestyle='--', label='Media + 3*STD')
        ax.legend()
    ax.set_title(titulo or f'Distribucion de {col_ratio}')
    return ax


def graficar_ratio_vs_exposicion(df, col_ratio='RATIO_FACTOR', col_exposicion='RATIO_EXPOSICION',
                                  col_color=None, limite_ratio=None):
    """
    Scatter de col_ratio vs. col_exposicion (deteccion visual de posibles
    atipicos, ver limpiar_datos.marcar_posibles_atipicos_ratio). Si se pasa
    col_color (ej. RATIO_PRECIO), colorea los puntos segun esa variable.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if col_color is not None:
        disp = ax.scatter(df[col_ratio], df[col_exposicion], c=df[col_color],
                           cmap='viridis', alpha=0.6, s=50)
        fig.colorbar(disp, ax=ax, label=col_color)
    else:
        ax.scatter(df[col_ratio], df[col_exposicion], alpha=0.6)

    if limite_ratio is not None:
        ax.axvline(x=limite_ratio, linestyle='--', color='black', label='Limite RATIO')
    ax.axhline(y=1, linestyle=':', color='gray', label='Exposicion sin cambio (1.0)')
    ax.set_xlabel(f'{col_ratio} (demanda destino / origen)')
    ax.set_ylabel(f'{col_exposicion} (destino / origen)')
    ax.set_title(f'{col_ratio} vs {col_exposicion}')
    ax.legend()
    return ax


def graficar_variables_vs_target(df_long, variables, y_var, tipo='continuas', col_split='split'):
    """
    FacetGrid de cada variable en 'variables' contra y_var: tipo='continuas'
    hace scatter+regresion, tipo='discretas' hace boxplot por col_split
    (Train/Test). df_long ya debe venir en formato largo (una fila por
    variable-valor, ver pd.DataFrame.melt) o se puede pasar el ancho y la
    funcion arma el melt.
    """
    if 'variable' not in df_long.columns:
        df_long = df_long.melt(id_vars=[y_var, col_split], value_vars=variables,
                                var_name='variable', value_name='valor').dropna(subset=['valor'])

    if tipo == 'continuas':
        g = sns.FacetGrid(df_long, col='variable', hue=col_split, col_wrap=4,
                           height=4, aspect=1.2, sharey=False, sharex=False)
        g.map(sns.regplot, 'valor', y_var, scatter_kws={'alpha': 0.2, 's': 15},
              line_kws={'linewidth': 1.5})
    else:
        g = sns.FacetGrid(df_long, col='variable', col_wrap=4, height=4, aspect=1.2,
                           sharey=False, sharex=False)
        g.map_dataframe(sns.boxplot, x='valor', y=y_var, hue=col_split)

    g.set_titles('{col_name}')
    g.add_legend()
    g.figure.suptitle(f'{tipo.capitalize()} vs {y_var}', y=1.02)
    return g


def graficar_importancia_rf(modelo_rf, nombres_variables, titulo='Importancia RF'):
    """Barh horizontal de feature_importances_ de un RandomForest ya entrenado."""
    fig, ax = plt.subplots()
    pd.Series(modelo_rf.feature_importances_, index=nombres_variables).sort_values().plot(
        kind='barh', title=titulo, ax=ax
    )
    ax.axvline(0, color='black', linewidth=0.8)
    return ax


def graficar_evolucion_ratio_precio(df, col_camp='CAMP', col_ratio='RATIO_PRECIO', step=5):
    """Evolucion de la media (+-1 std) de col_ratio por campana, para ver tendencia en el tiempo."""
    resumen = (df.groupby(col_camp)[col_ratio].agg(['mean', 'std', 'count'])
               .reset_index().sort_values(col_camp))
    resumen[col_camp] = resumen[col_camp].astype(str)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(data=resumen, x=col_camp, y='mean', marker='o', linewidth=2,
                 markersize=6, label=f'Media de {col_ratio}', color='steelblue', ax=ax)
    ax.fill_between(range(len(resumen)), resumen['mean'] - resumen['std'], resumen['mean'] + resumen['std'],
                     alpha=0.3, color='steelblue', label='+/-1 Std Dev')

    xtick_positions = range(0, len(resumen), step)
    ax.set_xticks(list(xtick_positions))
    ax.set_xticklabels(resumen[col_camp].values[::step], rotation=45)
    ax.set_xlabel('Campana')
    ax.set_ylabel(f'{col_ratio} promedio')
    ax.set_title(f'Evolucion de {col_ratio} en el tiempo')
    ax.legend()
    return ax


def graficar_learning_curve_temporal(df, pipeline, features_num, features_cat, col_camp='CAMP', target_col='LOG_RATIO_FACTOR', cortes_candidatos=None, metric_func=None):
    """
    Evalua un pipeline en distintos puntos de corte temporal (Train/Test) para
    graficar una learning curve. Ayuda a decidir el CORTE_TRAIN_TEST optimo.
    
    df: DataFrame procesado listo para modelar.
    pipeline: Objeto Pipeline de scikit-learn (sin entrenar).
    features_num, features_cat: Listas de variables a usar.
    cortes_candidatos: Lista de campanas (int) a evaluar como punto de corte.
    metric_func: Opcional. Funcion que reciba (y_true, y_pred) y devuelva un numero. Por defecto usa R2.
    """
    df = df.copy()
    df[col_camp] = df[col_camp].astype(int)
    
    if cortes_candidatos is None:
        # Por defecto, toma las campanas del medio (del 40% al 90% de la linea de tiempo)
        campanas = sorted(df[col_camp].unique())
        cortes_candidatos = campanas[int(len(campanas)*0.4) : int(len(campanas)*0.9)]
        
    resultados = []
    
    for corte in cortes_candidatos:
        train = df[df[col_camp] < corte].copy()
        test = df[df[col_camp] >= corte].copy()
        
        if len(train) == 0 or len(test) == 0:
            continue
            
        X_train = train[features_num + features_cat]
        y_train = train[target_col]
        X_test = test[features_num + features_cat]
        y_test = test[target_col]
        
        # Entrenar modelo
        pipeline.fit(X_train, y_train)
        
        # Predecir y evaluar
        y_pred = pipeline.predict(X_test)
        
        if metric_func:
            score = metric_func(y_test, y_pred)
        else:
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
            
        resultados.append({
            'Corte': str(corte),
            'Test_Score': score,
            'Train_Size': len(train),
            'Test_Size': len(test)
        })
        
    res_df = pd.DataFrame(resultados)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=res_df, x='Corte', y='Test_Score', marker='o', linewidth=2, color='darkorange', ax=ax)
    ax.set_title('Learning Curve Temporal: Estabilidad del Modelo por Corte')
    ax.set_xlabel('Campaña de Corte (Todo lo >= a esta campaña es Test)')
    ax.set_ylabel('Score en Test (R2 o Métrica Personalizada)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    return ax, res_df

