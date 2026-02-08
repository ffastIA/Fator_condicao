import cv2
import numpy as np
import os

# --- DADOS CONHECIDOS (GABARITO) ---
NOME_ARQUIVO = r"C:\Users\usuario\OneDrive - Engine Tecnologia\Projetos\Python\Aquicultura\FatorCondicao\imagens\Lateral_sem_fundo.png"
COMPRIMENTO_REAL_CM = 38.0  # O valor que você afirmou ser a verdade


def calibrar_e_medir():
    if not os.path.exists(NOME_ARQUIVO):
        print(f"❌ Erro: O arquivo '{NOME_ARQUIVO}' não foi encontrado.")
        return

    # 1. Carregar e Processar
    img = cv2.imread(NOME_ARQUIVO)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. Encontrar Peixe (Hull)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Erro: Nenhum contorno detectado.")
        return

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)  # Usando Hull para garantir solidez

    # 3. Extrair Medidas em PIXELS
    rect = cv2.minAreaRect(hull)
    (x, y), (w, h), angle = rect

    # O maior lado do retângulo é sempre o comprimento do peixe
    comprimento_px = max(w, h)
    largura_px = min(w, h)
    area_px = cv2.contourArea(hull)

    # 4. CÁLCULO REVERSO DO FATOR
    # Se 25cm equivalem a X pixels, então 1cm equivale a X/25 pixels.
    fator_calibrado = comprimento_px / COMPRIMENTO_REAL_CM

    # 5. Calcular as outras medidas usando o fator descoberto
    largura_cm = largura_px / fator_calibrado
    area_cm2 = area_px / (fator_calibrado ** 2)

    # --- RELATÓRIO NO TERMINAL ---
    print(f"\n=== CALIBRAÇÃO UNITÁRIA: {NOME_ARQUIVO} ===")
    print(f"Gabarito Fornecido: {COMPRIMENTO_REAL_CM} cm\n")

    print("--- 1. DESCOBERTA DO FATOR ---")
    print(f"Comprimento Detectado: {comprimento_px:.2f} pixels")
    print(f"Cálculo: {comprimento_px:.2f} px / {COMPRIMENTO_REAL_CM} cm")
    print(f"👉 FATOR DE CONVERSÃO EXATO: {fator_calibrado:.4f} px/cm")

    print("\n--- 2. MEDIDAS CALCULADAS (Check) ---")
    print(f"Largura:  {largura_px:.0f} px  -> {largura_cm:.2f} cm")
    print(f"Área:     {area_px:.0f} px² -> {area_cm2:.2f} cm²")

    # --- GERAR IMAGEM PROVA ---
    img_out = img.copy()
    box = np.int32(cv2.boxPoints(rect))

    # Desenha Hull (Azul Claro Preenchido)
    overlay = img.copy()
    cv2.drawContours(overlay, [hull], -1, (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.4, img_out, 0.6, 0, img_out)

    # Desenha Box (Vermelho)
    cv2.drawContours(img_out, [box], 0, (0, 0, 255), 2)

    # Escreve os dados
    txt_fator = f"FATOR DESCOBERTO: {fator_calibrado:.2f} px/cm"
    txt_area = f"AREA REAL: {area_cm2:.2f} cm2"

    cv2.putText(img_out, txt_fator, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_out, txt_area, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite("resultado_calibracao_img0207.jpg", img_out)
    print("\n✅ Imagem salva como: resultado_calibracao_img0207.jpg")


if __name__ == "__main__":
    calibrar_e_medir()