import openvino as ov
import tensorflow as tf
import os
import shutil

# Chemins
model_path = "models/saved_model_temp"
output_dir = "models/export_npu"  # On crée un sous-dossier dédié

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("--- Chargement et Conversion ---")
try:
    # Conversion
    ov_model = ov.convert_model(model_path)

    # Sauvegarde avec un nom unique pour éviter les conflits
    xml_path = os.path.join(output_dir, "sleep_model_npu.xml")

    print("--- Écriture des fichiers IR (FP16) ---")
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)

    print(f"✅ SUCCÈS !")
    print(f"Fichiers créés dans : {output_dir}")

except Exception as e:
    print(f"❌ Erreur : {e}")