import cv2
import numpy as np
import csv
import os
import sys

# --- Configurações ---
EXTENSOES_IMAGEM = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
NOME_CSV = "relatorio_final_completo.csv"


def desenhar_overlay(imagem, hull, box, metricas):
    """ Desenha visualização usando o Convex Hull (Área Sólida) """
    img_final = imagem.copy()
    overlay = imagem.copy()

    # Desenha o Hull preenchido (Ciano) - Isso representa a área do peixe
    cv2.drawContours(overlay, [hull], -1, (255, 255, 0), -1)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_final, 1 - alpha, 0, img_final)

    # Desenha Bounding Box (Vermelho) - Representa a área do Box
    cv2.drawContours(img_final, [box], 0, (0, 0, 255), 2)

    # Textos
    texto = f"Peixe: {metricas['hull_area_cm2']:.1f} cm2 | Box: {metricas['box_area_cm2']:.1f} cm2"
    cv2.putText(img_final, texto, (int(box[0][0]), int(box[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img_final


def processar_imagem_hull(caminho_img, fator_px_cm):
    try:
        img = cv2.imread(caminho_img)
        if img is None: return False, "Erro leitura", None, None

        # 1. Pré-proc
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        # 2. Segmentação
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Morfologia para reduzir ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Salva binário para debug
        img_binaria = thresh.copy()

        # 4. Contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False, "Zero contornos encontrados", None, img_binaria

        # Pega maior contorno
        cnt = max(contours, key=cv2.contourArea)

        # 5. Convex Hull
        hull = cv2.convexHull(cnt)

        # --- CÁLCULOS GEOMÉTRICOS ---

        # A. Do Peixe (Hull)
        hull_area_px = cv2.contourArea(hull)

        # B. Do Bounding Box (Retângulo)
        rect = cv2.minAreaRect(hull)
        (x, y), (w, h), angle = rect

        # Dimensões lineares
        box_comp_px = max(w, h)
        box_larg_px = min(w, h)

        # Área do Box (Base x Altura)
        box_area_px = box_comp_px * box_larg_px

        # --- CONVERSÃO PARA CM ---

        # Fator Linear
        box_comp_cm = box_comp_px / fator_px_cm
        box_larg_cm = box_larg_px / fator_px_cm

        # Fator Quadrático (Área)
        fator_area = (fator_px_cm ** 2)
        hull_area_cm2 = hull_area_px / fator_area
        box_area_cm2 = box_area_px / fator_area

        metricas = {
            # CM
            "hull_area_cm2": hull_area_cm2,
            "box_comp_cm": box_comp_cm,
            "box_larg_cm": box_larg_cm,
            "box_area_cm2": box_area_cm2,
            # PIXELS
            "hull_area_px": hull_area_px,
            "box_comp_px": box_comp_px,
            "box_larg_px": box_larg_px,
            "box_area_px": box_area_px
        }

        # Visualização
        box = np.int32(cv2.boxPoints(rect))
        img_out = desenhar_overlay(img, hull, box, metricas)

        return True, metricas, img_out, img_binaria

    except Exception as e:
        return False, f"EXCEÇÃO: {str(e)}", None, None


def obter_input_limpo(msg):
    return input(msg).strip().replace('"', '').replace("'", "")


def main():
    print("\n=== BIOMETRIA TILÁPIA (Relatório Completo) ===")

    dir_in = obter_input_limpo("1. Pasta FOTOS:\n>> ")
    if not os.path.exists(dir_in):
        print("Erro: Pasta de entrada não existe.");
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

        # --- CABEÇALHO ATUALIZADO ---
        writer.writerow([
            "Arquivo", "Status",
            "Box_Comp_cm", "Box_Larg_cm", "Box_Area_cm2", "Peixe_Area_cm2",  # Métricas Reais
            "Box_Comp_px", "Box_Larg_px", "Box_Area_px", "Peixe_Area_px",  # Métricas Pixels
            "Fator"
        ])

        for nome in arquivos:
            path_in = os.path.join(dir_in, nome)
            status, dados, img_out, img_bin = processar_imagem_hull(path_in, fator)

            if status:
                cv2.imwrite(os.path.join(dir_out, f"RESULT_{nome}"), img_out)
                cv2.imwrite(os.path.join(dir_out, f"BIN_{nome}"), img_bin)

                # --- GRAVAÇÃO DOS DADOS ---
                writer.writerow([
                    nome, "OK",
                    # CM
                    f"{dados['box_comp_cm']:.2f}",
                    f"{dados['box_larg_cm']:.2f}",
                    f"{dados['box_area_cm2']:.2f}",
                    f"{dados['hull_area_cm2']:.2f}",
                    # PX
                    f"{dados['box_comp_px']:.2f}",
                    f"{dados['box_larg_px']:.2f}",
                    f"{dados['box_area_px']:.0f}",
                    f"{dados['hull_area_px']:.0f}",
                    # Meta
                    fator
                ])
                print(f"✅ {nome}: Peixe {dados['hull_area_cm2']:.1f} cm² | Box {dados['box_area_cm2']:.1f} cm²")
            else:
                writer.writerow([nome, f"ERRO: {dados}", 0, 0, 0, 0, 0, 0, 0, 0, fator])
                print(f"❌ {nome}: {dados}")


if __name__ == "__main__":
    main()