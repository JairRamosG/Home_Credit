# Tipos de Encoding para Variables Categóricas

## ¿Por qué encodear?

Los modelos numéricos no entienden texto. Necesitamos convertir categorías como `"Working"`, `"Commercial associate"`, `"Pensioner"` en números.

---

## 1. Label Encoding

**Cómo funciona:** Asigna un número entero a cada categoría.

```
Working            → 0
Commercial associate → 1
Pensioner          → 2
State servant      → 3
```

**Cuándo usarlo:**
- Variables **binarias** (sí/no, hombre/mujer)
- Variables **ordinales** donde el orden importa (bajo/medio/alto)

**Problema:** Si la variable NO tiene orden natural, el modelo interpreta que `3 > 2 > 1 > 0` y eso es mentira. `Pensioner` no es "mayor" que `Working`.

---

## 2. One-Hot Encoding

**Cómo funciona:** Crea una columna binaria por cada categoría.

```
                    Working  Commercial associate  Pensioner  State servant
Working                   1                    0          0              0
Commercial associate      0                    1          0              0
Pensioner                 0                    0          1              0
State servant             0                    0          0              1
```

**Cuándo usarlo:**
- Variables categóricas **sin orden natural** (tipo de ingreso, ocupación)
- Pocas categorías (< 15)

**Problema:**
- Si hay 50 categorías, creás 50 columnas nuevas → **explosión de dimensionalidad**
- Alta **multicolinealidad** (las columnas se correlacionan entre sí)
- Solución al multicolinealidad: `drop_first=True` (quita una columna de referencia)

---

## 3. Target Encoding (Mean Encoding)

**Cómo funciona:** Reemplaza cada categoría con el promedio de la TARGET para esa categoría.

```
Working            → 0.067  (promedio de default en Working)
Commercial associate → 0.074
Pensioner          → 0.057
State servant      → 0.058
```

**Cuándo usarlo:**
- Variables con **muchas categorías** (organización, ocupación, etc.)
- Cuando one-hot crearía demasiadas columnas

**Problema:** **Data leakage** — si calculás el promedio con los mismos datos que usás para entrenar, el modelo "hace trampa". Solución: calcular el target encoding SOLO en el split de entrenamiento.

---

## 4. Frequency Encoding

**Cómo funciona:** Reemplaza cada categoría con su frecuencia relativa.

```
Working            → 0.518  (51.8% de los clientes son Working)
Commercial associate → 0.233
Pensioner          → 0.178
State servant      → 0.071
```

**Cuándo usarlo:**
- Cuando la **frecuencia** de la categoría importa (ej: clientes comunes vs raros)
- Variable con muchas categorías

**Problema:** Dos categorías con la misma frecuencia reciben el mismo valor, aunque impacten distinto en la TARGET.

---

## 5. Ordinal Encoding

**Cómo funciona:** Asigna números basados en un orden predefinido por vos.

```
Low      → 0
Medium   → 1
High     → 2
```

**Cuándo usarlo:**
- Variables **ordinales explícitas** (nivel de educación, nivel de riesgo, calificación)
- Cuando el orden es semántico, no arbitrario

---

## 6. Binary Encoding

**Cómo funciona:** Convierte cada categoría a binario, luego crea columnas por cada bit.

```
Categoría A (1) → 001 → columnas: bit1=0, bit2=0, bit3=1
Categoría B (2) → 010 → columnas: bit1=0, bit2=1, bit3=0
Categoría C (3) → 011 → columnas: bit1=0, bit2=1, bit3=1
Categoría D (4) → 100 → columnas: bit1=1, bit2=0, bit3=0
```

**Cuándo usarlo:**
- Variables con **muchas categorías** (50+)
- Cuando querés menos columnas que one-hot

**Problema:** Los bits no tienen significado semántico.

---

## 7. Hashing Encoding

**Cómo funciona:** Aplica una función hash a cada categoría y usa el resultado como número fijo.

**Cuándo usarlo:**
- Variables con **extremadamente muchas categorías** (millones)
- Cuando no podés guardar el mapeo completo

---

## Tabla resumen

| Método | Columnas creadas | Orden natural | Riesgo leakage | Mejor para |
|---|---|---|---|---|
| Label Encoding | 1 | Sí (forzado) | No | Binarias, ordinales |
| One-Hot | N categorías | No | No | Pocas categorías (<15) |
| Target Encoding | 1 | No | **Sí** | Muchas categorías |
| Frequency Encoding | 1 | No | No | Categorías raras |
| Ordinal Encoding | 1 | Sí | No | Ordinales explícitas |
| Binary Encoding | log2(N) | No | No | Muchas categorías |
| Hashing | Fijo | No | No | Categorías extremas |

---

## ¿Cuál elegir para nuestro proyecto?

### Variables binarias (FLAG_OWN_CAR, CODE_GENDER, etc.)
→ **Label Encoding** (ya lo hacemos, está bien)

### Variables con pocas categorías (<10)
→ **One-Hot Encoding** (ya lo hacemos, está bien)

### Variables con MUCHAS categorías (>15)
Estas son las problemáticas en nuestro dataset:
- `ORGANIZATION_TYPE` → 58 categorías
- `OCCUPATION_TYPE` → 18 categorías
- `NAME_INCOME_TYPE` → 8 categorías (ok con one-hot)
- `NAME_EDUCATION_TYPE` → 5 categorías (ok con one-hot)

Para `ORGANIZATION_TYPE` y `OCCUPATION_TYPE`, las opciones son:
1. **Target Encoding** — reemplazar por el promedio de la TARGET
2. **Frequency Encoding** — reemplazar por la frecuencia
3. **Agrupar categorías** — reducir a grupos significativos antes de one-hot

### Recomendación para este proyecto:

| Variable | Estrategia sugerida | Razón |
|---|---|---|
| Binarias (6 cols) | Label Encoding | Solo 2 valores |
| NAME_INCOME_TYPE | One-Hot | 8 categorías, manejable |
| NAME_EDUCATION_TYPE | One-Hot | 5 categorías, manejable |
| NAME_FAMILY_STATUS | One-Hot | 6 categorías, manejable |
| NAME_HOUSING_TYPE | One-Hot | 6 categorías, manejable |
| OCCUPATION_TYPE | **Target Encoding** | 18 categorías, una sola columna |
| ORGANIZATION_TYPE | **Target Encoding** | 58 categorías, explosión con one-hot |
| Resto multi-categoría | One-Hot | Pocas categorías |
