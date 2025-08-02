
# Trabalho de Estatística - ENEM 2024
# Autor: Philipe Fransozi
# Repositório: https://github.com/pFransozi/mestrado-estatistica-trabalho

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Leitura dos dados
df = pd.read_csv('DADOS/PARTICIPANTES_2024.csv', encoding='latin1', sep=';')

# 2. Seleção de colunas relevantes
df = df[['TP_DEPENDENCIA_ADM_ESC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']].copy()

# 3. Mapeamento do tipo de escola
df['TIPO_ESCOLA'] = df['TP_DEPENDENCIA_ADM_ESC'].map({
    1: 'Pública',   # Federal
    2: 'Pública',   # Estadual
    3: 'Pública',   # Municipal
    4: 'Privada'    # Privada
})

# 4. Remoção de valores ausentes
df = df.dropna()

# 5. Análise descritiva
resumo_estatistico = df.groupby('TIPO_ESCOLA')[['NU_NOTA_MT', 'NU_NOTA_REDACAO']].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
resumo_estatistico.to_csv('tabela_resumo.csv')

# 6. Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='TIPO_ESCOLA', y='NU_NOTA_MT', data=df)
plt.title('Distribuição das Notas de Matemática por Tipo de Escola')
plt.xlabel('Tipo de Escola')
plt.ylabel('Nota de Matemática')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('boxplot_matematica_tipo_escola.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='TIPO_ESCOLA', y='NU_NOTA_REDACAO', data=df)
plt.title('Distribuição das Notas de Redação por Tipo de Escola')
plt.xlabel('Tipo de Escola')
plt.ylabel('Nota de Redação')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('boxplot_redacao_tipo_escola.png')
plt.close()

# 7. Regressão linear simples
x = df['NU_NOTA_MT']
y = df['NU_NOTA_REDACAO']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Equação da reta: Ŷ = {intercept:.2f} + {slope:.2f}x")
print(f"Coeficiente de determinação R² = {r_value**2:.4f}")
print(f"Valor-p do coeficiente angular: {p_value:.4f}")

# Gráfico de regressão
plt.figure(figsize=(10, 6))
sns.regplot(x='NU_NOTA_MT', y='NU_NOTA_REDACAO', data=df, scatter_kws={'alpha':0.3})
plt.title('Relação entre Notas de Matemática e Redação')
plt.xlabel('Nota de Matemática')
plt.ylabel('Nota de Redação')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('regressao_redacao_matematica.png')
plt.close()

# 8. Teste t de Student
privada = df[df['TIPO_ESCOLA'] == 'Privada']['NU_NOTA_REDACAO']
publica = df[df['TIPO_ESCOLA'] == 'Pública']['NU_NOTA_REDACAO']
t_stat, p_val = stats.ttest_ind(privada, publica, equal_var=False)
print(f"t = {t_stat:.2f}, p = {p_val:.4f}")

# 9. Teste de Mann-Whitney
u_stat, p_mw = stats.mannwhitneyu(privada, publica, alternative='two-sided')
print(f"U = {u_stat:.2f}, p = {p_mw:.4f}")
