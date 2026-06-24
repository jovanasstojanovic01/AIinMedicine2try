import sys
import time

import extract_grape_features
import merge_grape_data
import create_gru_sequences
import train

def main():
    print("=================================================================")
    print("      SISTEM ZA MULTIMODALNU PREDIKCIJU PROGRESIJE GLAUKOMA      ")
    print("=================================================================\n")
    
    start_time = time.time()

    print("[KORAK 1/4] Pokretanje UNet ekstrakcije geometrijskih parametara...")
    try:
        extract_grape_features.main()
    except Exception as e:
        print(f"\n[KRIZNA GREŠKA] Korak 1 je pukao: {str(e)}")
        sys.exit(1)

    print("\n" + "-"*50 + "\n")

    print("[KORAK 2/4] Pokretanje spajanja tabelarnih podataka (Pandas Merge)...")
    try:
        merge_grape_data.main()
    except Exception as e:
        print(f"\n[KRIZNA GREŠKA] Korak 2 je pukao: {str(e)}")
        sys.exit(1)

    print("\n" + "-"*50 + "\n")

    print("[KORAK 3/4] Pokretanje kreiranja 3D hronoloških sekvenci za GRU...")
    try:
        create_gru_sequences.main()
    except Exception as e:
        print(f"\n[KRIZNA GREŠKA] Korak 3 je pukao: {str(e)}")
        sys.exit(1)

    print("\n" + "-"*50 + "\n")

    print("[KORAK 4/4] Pokretanje obuke duboke GRU mreže...")
    try:
        train.main()
    except Exception as e:
        print(f"\n[KRIZNA GREŠKA] Korak 4 je pukao: {str(e)}")
        sys.exit(1)

    total_time = time.time() - start_time
    print("\n=================================================================")
    print("   CELOKUPAN PIPELINE JE USPEŠNO ZAVRŠEN! 🎉")
    print(f"   Ukupno vreme izvršavanja: {total_time/60:.2f} minuta.")
    print("   Svi modeli, tabele i grafikoni se nalaze u 'checkpoints/' i 'outputs/'.")
    print("=================================================================")

if __name__ == "__main__":
    main()