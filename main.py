import cv2
import numpy as np
import csv
import os
import sys

# --- Configurações ---
EXTENSOES_IMAGEM = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
NOME_CSV = "relatorio_final_completo.csv"


def desenhar_overlay(imagem, hull, box, metricas):
    """
    Desenha visualização com:
    - Texto centralizado no box
    - Proteção para não sair da imagem (bordas)
    - Posicionamento inteligente (topo ou base dependendo do espaço)
    """
    img_final = imagem.copy()
    overlay = imagem.copy()

    # Obtém dimensões da imagem para verificação de bordas
    h_img, w_img = imagem.shape[:2]

    # 1. Desenha Hull (Área do Peixe)
    cv2.drawContours(overlay, [hull], -1, (255, 255, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_final, 1 - alpha, 0, img_final)

    # 2. Desenha Box (Área do Retângulo)
    cv2.drawContours(img_final, [box], 0, (0, 0, 255), 2)

    # --- LÓGICA DE TEXTO INTELIGENTE ---

    linhas_texto = [
        f"Peixe: {metricas['hull_area_cm2']:.1f} cm2 | Box: {metricas['box_area_cm2']:.1f} cm2",
        f"Dim (cm): {metricas['box_comp_cm']:.1f} x {metricas['box_larg_cm']:.1f} cm",
        f"Dim (px): {metricas['box_comp_px']:.0f} x {metricas['box_larg_px']:.0f} px"
    ]

    # Calcula limites do box
    box_x_coords = box[:, 0]
    box_y_coords = box[:, 1]

    # Ponto X Central do Box (para centralizar o texto)
    center_x = int(np.mean(box_x_coords))

    # Pontos Y Extremos
    min_y = int(np.min(box_y_coords))  # Topo do peixe
    max_y = int(np.max(box_y_coords))  # Base do peixe

    # Decisão: Escrever em cima ou embaixo?
    # Se tiver menos de 80px de espaço no topo, escreve embaixo.
    altura_bloco_texto = len(linhas_texto) * 25
    if min_y < altura_bloco_texto + 10:
        y_inicial = max_y + 20  # Escreve ABAIXO
        direcao = 1  # Texto desce
    else:
        y_inicial = min_y - 10  # Escreve ACIMA (padrão)
        direcao = -1  # Texto sobe (para a lista invertida funcionar, ajustamos no loop)

    # Configuração da Fonte
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.5
    espessura = 1
    padrao_linha = 25

    # Se a direção for "subindo" (texto acima), invertemos a lista para desenhar da base para o topo
    lista_para_desenhar = reversed(linhas_texto) if direcao == -1 else linhas_texto

    y_atual = y_inicial

    for texto in lista_para_desenhar:
        # Calcula tamanho do texto
        (w_txt, h_txt), baseline = cv2.getTextSize(texto, fonte, escala, espessura)

        # 1. Centralização: Pega o centro do box e subtrai metade do texto
        x_txt = center_x - (w_txt // 2)

        # 2. Proteção de Bordas (Clamping):
        # Garante margem mínima de 10px na esquerda
        if x_txt < 10:
            x_txt = 10
        # Garante margem mínima de 10px na direita
        elif (x_txt + w_txt) > w_img - 10:
            x_txt = w_img - w_txt - 10

        # Desenha Fundo Preto (Background)
        # Ajusta coordenadas do retângulo de fundo
        cv2.rectangle(img_final,
                      (x_txt - 4, y_atual - h_txt - 4),
                      (x_txt + w_txt + 4, y_atual + baseline + 4),
                      (0, 0, 0), -1)

        # Escreve o Texto
        cv2.putText(img_final, texto, (x_txt, y_atual), fonte, escala, (255, 255, 255), espessura, cv2.LINE_AA)

        # Atualiza Y para a próxima linha
        if direcao == -1:
            y_atual -= padrao_linha  # Sobe
        else:
            y_atual += padrao_linha  # Desce

    return img_final


def processar_imagem_hull(caminho_img, fator_px_cm):
    try:
        img = cv2.imread(caminho_img)
        if img is None: return False, "Erro leitura", None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        img_binaria = thresh.copy()

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False, "Zero contornos", None, img_binaria

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        # Cálculos
        hull_area_px = cv2.contourArea(hull)
        rect = cv2.minAreaRect(hull)
        (x, y), (w, h), angle = rect

        box_comp_px = max(w, h)
        box_larg_px = min(w, h)
        box_area_px = box_comp_px * box_larg_px

        box_comp_cm = box_comp_px / fator_px_cm
        box_larg_cm = box_larg_px / fator_px_cm
        fator_area = (fator_px_cm ** 2)
        hull_area_cm2 = hull_area_px / fator_area
        box_area_cm2 = box_area_px / fator_area

        metricas = {
            "hull_area_cm2": hull_area_cm2,
            "box_comp_cm": box_comp_cm,
            "box_larg_cm": box_larg_cm,
            "box_area_cm2": box_area_cm2,
            "hull_area_px": hull_area_px,
            "box_comp_px": box_comp_px,
            "box_larg_px": box_larg_px,
            "box_area_px": box_area_px
        }

        box = np.int32(cv2.boxPoints(rect))
        img_out = desenhar_overlay(img, hull, box, metricas)

        return True, metricas, img_out, img_binaria

    except Exception as e:
        return False, f"EXCEÇÃO: {str(e)}", None, None


def obter_input_limpo(msg):
    entrada = input(msg).strip()
    return entrada.replace('"', '').replace("'", "")


def main():
    print("\n=== BIOMETRIA TILÁPIA (Texto Centralizado e Seguro) ===")

    dir_in = obter_input_limpo("1. Pasta FOTOS:\n>> ")

    if os.path.isfile(dir_in):
        print("⚠️  Arquivo detectado. Usando diretório pai.")
        dir_in = os.path.dirname(dir_in)

    if not os.path.isdir(dir_in):
        print(f"❌ Erro: Caminho inválido: {dir_in}")
        return

    dir_out = obter_input_limpo("\n2. Pasta SAÍDA:\n>> ")

    try:
        fator = float(input("\n3. Fator (px/cm) [Ex: 75.87]:\n>> ").replace(',', '.'))
    except:
        print("Fator inválido.");
        return

    os.makedirs(dir_out, exist_ok=True)
    arquivos = [f for f in os.listdir(dir_in) if f.lower().endswith(EXTENSOES_IMAGEM)]

    print(f"\nIniciando em {len(arquivos)} imagens...")

    path_csv = os.path.join(dir_out, NOME_CSV)

    with open(path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Arquivo", "Status",
            "Box_Comp_cm", "Box_Larg_cm", "Box_Area_cm2", "Peixe_Area_cm2",
            "Box_Comp_px", "Box_Larg_px", "Box_Area_px", "Peixe_Area_px",
            "Fator"
        ])

        for nome in arquivos:
            path_in = os.path.join(dir_in, nome)
            status, dados, img_out, img_bin = processar_imagem_hull(path_in, fator)

            if status:
                cv2.imwrite(os.path.join(dir_out, f"RESULT_{nome}"), img_out)
                cv2.imwrite(os.path.join(dir_out, f"BIN_{nome}"), img_bin)

                writer.writerow([
                    nome, "OK",
                    f"{dados['box_comp_cm']:.2f}",
                    f"{dados['box_larg_cm']:.2f}",
                    f"{dados['box_area_cm2']:.2f}",
                    f"{dados['hull_area_cm2']:.2f}",
                    f"{dados['box_comp_px']:.2f}",
                    f"{dados['box_larg_px']:.2f}",
                    f"{dados['box_area_px']:.0f}",
                    f"{dados['hull_area_px']:.0f}",
                    fator
                ])
                print(f"✅ {nome}: Peixe {dados['hull_area_cm2']:.1f} cm²")
            else:
                writer.writerow([nome, f"ERRO: {dados}", 0, 0, 0, 0, 0, 0, 0, 0, fator])
                print(f"❌ {nome}: {dados}")


if __name__ == "__main__":
    main()