import cv2
import numpy as np
import csv
import os
import sys

# --- Configurações ---
EXTENSOES_IMAGEM = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
NOME_CSV = "relatorio_diagnostico.csv"


def processar_imagem_hull(caminho_img, fator_px_cm):
    try:
        print(f"   > Lendo: {os.path.basename(caminho_img)}...")
        img = cv2.imread(caminho_img)
        if img is None:
            return False, "cv2.imread retornou None (arquivo corrompido ou caminho errado)", None, None

        # Processamento (Convex Hull)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        img_binaria = thresh.copy()

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "Zero contornos encontrados (imagem toda preta?)", None, img_binaria

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        area_px = cv2.contourArea(hull)
        rect = cv2.minAreaRect(hull)
        (x, y), (w, h), angle = rect

        area_cm2 = area_px / (fator_px_cm ** 2)
        comp_cm = max(w, h) / fator_px_cm
        larg_cm = min(w, h) / fator_px_cm

        metricas = {
            "area_cm2": area_cm2,
            "comp_cm": comp_cm,
            "larg_cm": larg_cm,
            "area_px": area_px
        }

        # Desenho
        img_out = img.copy()
        cv2.drawContours(img_out, [hull], -1, (255, 255, 0), -1)
        cv2.addWeighted(img_out, 0.4, img, 0.6, 0, img_out)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(img_out, [box], 0, (0, 0, 255), 2)

        return True, metricas, img_out, img_binaria

    except Exception as e:
        return False, f"EXCEÇÃO PYTHON: {str(e)}", None, None


def obter_input_limpo(msg):
    return input(msg).strip().replace('"', '').replace("'", "")


def main():
    print("\n=== DIAGNÓSTICO DE SISTEMA DE VISÃO ===")

    # 1. Validação de Diretórios com Logs Absolutos
    dir_in = obter_input_limpo("1. Pasta FOTOS:\n>> ")
    abs_in = os.path.abspath(dir_in)
    print(f"   [INFO] Caminho absoluto entrada: {abs_in}")

    if not os.path.exists(abs_in):
        print("   [ERRO CRÍTICO] A pasta de entrada NÃO EXISTE neste caminho.")
        return

    dir_out = obter_input_limpo("\n2. Pasta SAÍDA:\n>> ")
    abs_out = os.path.abspath(dir_out)
    print(f"   [INFO] Caminho absoluto saída: {abs_out}")

    try:
        os.makedirs(abs_out, exist_ok=True)
        print("   [OK] Pasta de saída validada.")
    except Exception as e:
        print(f"   [ERRO CRÍTICO] Não foi possível criar pasta de saída: {e}")
        return

    # Fator fixo ou input
    try:
        fator = float(input("\n3. Fator (px/cm) [Ex: 45.5]:\n>> ").replace(',', '.'))
    except:
        fator = 1.0
        print("   [AVISO] Fator inválido. Usando 1.0 para teste.")

    # 2. Listagem de Arquivos
    todos_arquivos = os.listdir(abs_in)
    print(f"\n[DIAGNÓSTICO] Total de arquivos na pasta: {len(todos_arquivos)}")

    arquivos_validos = [f for f in todos_arquivos if f.lower().endswith(EXTENSOES_IMAGEM)]
    print(f"[DIAGNÓSTICO] Imagens válidas (filtro .jpg/.png): {len(arquivos_validos)}")

    if len(arquivos_validos) == 0:
        print("   [ALERTA] Nenhuma imagem encontrada! Verifique se a extensão está correta.")
        print(f"   Exemplos encontrados na pasta: {todos_arquivos[:5]}")
        return

    # 3. Execução
    path_csv = os.path.join(abs_out, NOME_CSV)
    print(f"\n[INICIANDO] Tentando gravar CSV em: {path_csv}")

    try:
        with open(path_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Arquivo", "Status", "Area_cm2", "Msg_Erro"])
            print("   [OK] CSV criado com sucesso.")

            for i, nome in enumerate(arquivos_validos):
                path_in = os.path.join(abs_in, nome)

                print(f"\n--- Imagem {i + 1}/{len(arquivos_validos)}: {nome} ---")
                status, dados, img_out, img_bin = processar_imagem_hull(path_in, fator)

                if status:
                    # Tenta salvar imagens e verifica retorno do OpenCV
                    p_out = os.path.join(abs_out, f"RES_{nome}")
                    p_bin = os.path.join(abs_out, f"BIN_{nome}")

                    ok1 = cv2.imwrite(p_out, img_out)
                    ok2 = cv2.imwrite(p_bin, img_bin)

                    if ok1 and ok2:
                        print(f"   [SUCESSO] Imagens salvas e Área: {dados['area_cm2']:.2f}")
                        writer.writerow([nome, "OK", f"{dados['area_cm2']:.2f}", "-"])
                    else:
                        print(f"   [ERRO DISCO] OpenCV falhou ao salvar imagem em '{abs_out}'. Verifique permissões.")
                        writer.writerow([nome, "ERRO_GRAVACAO", 0, "Falha cv2.imwrite"])
                else:
                    print(f"   [FALHA LÓGICA] {dados}")
                    writer.writerow([nome, "FALHA_PROCESSAMENTO", 0, dados])

    except Exception as e:
        print(f"\n[ERRO FATAL NO LOOP] {str(e)}")

    print("\n=== FIM DO DIAGNÓSTICO ===")


if __name__ == "__main__":
    main()