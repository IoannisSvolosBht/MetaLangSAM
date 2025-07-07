import streamlit as st
import leafmap.foliumap as leafmap
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
import os
import time
import zipfile
import geopandas as gpd
import matplotlib.pyplot as plt

# Konfiguration
st.set_page_config(layout="wide")
st.title("KI-basierte Objekterkennung mit Meta's LangSAM")

# Session State für Persistenz
if 'results' not in st.session_state:
    st.session_state.results = None
if 'map_visible' not in st.session_state:
    st.session_state.map_visible = True

# --- 1. Karte initialisieren ---
if st.session_state.map_visible:
    m = leafmap.Map(center=[52.4658, 13.3825], zoom=18, height=700)
    m.add_basemap("SATELLITE")
    m.to_streamlit(height=500, key="main_map")

# --- 2. Sidebar für Einstellungen ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    process_type = st.radio(
        "Modus wählen:",
        ["Automatische Segmentierung", "Text-Prompt Suche"],
        index=0
    )

    if process_type == "Text-Prompt Suche":
        text_prompt = st.text_input("Suchbegriff (Englisch)", "tree")
        box_threshold = st.slider("Box Threshold", 0.0, 1.0, 0.24)
        text_threshold = st.slider("Text Threshold", 0.0, 1.0, 0.24)

# --- 3. Bounding Box Eingabe ---
st.header(" Bounding Box festlegen")
cols = st.columns(4)
with cols[0]: left = st.number_input("Left (West)", value=13.38000, format="%.5f")
with cols[1]: bottom = st.number_input("Bottom (Süd)", value=52.46436, format="%.5f")
with cols[2]: right = st.number_input("Right (Ost)", value=13.38320, format="%.5f")
with cols[3]: top = st.number_input("Top (Nord)", value=52.46720, format="%.5f")

bbox = [left, bottom, right, top]
st.info(f"Aktuelle Bounding Box: `{bbox}`")

# --- 4. Prozess-Starter ---
if st.button("Starte Segmentierung", type="primary"):
    with st.spinner("Lade Satellitenbild..."):
        tiff_path = "satellite.tif"
        mask_path = "masks.tif"
        vector_path = "masks.shp"
        
        # Alte Dateien löschen
        for file in [tiff_path, mask_path, vector_path]:
            if os.path.exists(file): os.remove(file)

        # Satellitenbild herunterladen
        leafmap.map_tiles_to_geotiff(
            output=tiff_path, bbox=bbox, zoom=18, source="Satellite", overwrite=True
        )

    if process_type == "Automatische Segmentierung":
        # --- Automatische Segmentierung ---
        with st.spinner("Segmentiere Objekte..."):
            start_time = time.time()
            sam = SamGeo(
                model_type="vit_h",
                sam_kwargs={
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.86,
                    "stability_score_thresh": 0.92
                }
            )
            sam.generate(tiff_path, output=mask_path, foreground=True)
            
            # Erstelle Visualisierung mit Farbpalette
            fig, ax = plt.subplots(figsize=(12, 12))
            sam.show_anns(
                cmap="Greens",
                box_color="red",
                title="Automatic Segmentation",
                blend=True,
                ax=ax
            )
            plt.axis('off')
            vis_path = "visualization.png"
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            duration = time.time() - start_time

        # Ergebnisse speichern
        st.session_state.results = {
            'tiff_path': tiff_path,
            'mask_path': mask_path,
            'vector_path': vector_path,
            'vis_path': vis_path,
            'process_type': process_type,
            'duration': duration
        }

    else:
        # --- Text-Prompt Segmentierung ---
        with st.spinner(f"Suche nach '{text_prompt}'..."):
            start_time = time.time()
            sam = LangSAM()
            sam.predict(
                image=tiff_path,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                output=mask_path
            )
            
            # Erstelle Visualisierung mit Farbpalette
            fig, ax = plt.subplots(figsize=(12, 12))
            sam.show_anns(
                cmap="Greens",
                box_color="red",
                title=f"Segmentation of {text_prompt}",
                blend=True,
                ax=ax
            )
            plt.axis('off')
            vis_path = "visualization.png"
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            duration = time.time() - start_time

        # Ergebnisse speichern
        st.session_state.results = {
            'tiff_path': tiff_path,
            'mask_path': mask_path,
            'vector_path': vector_path,
            'vis_path': vis_path,
            'process_type': process_type,
            'text_prompt': text_prompt,
            'duration': duration,
            'sam': sam
        }
    
    # Karte nach der Prozessierung ausblenden
    st.session_state.map_visible = False
    st.rerun()

# --- 5. Ergebnisse anzeigen (persistent) ---
if st.session_state.results:
    results = st.session_state.results
    
    st.success(f"Segmentierung fertig ({results['duration']:.1f}s)")
    st.header(f"Ergebnisse: {results['text_prompt'] if results['process_type'] == 'Text-Prompt Suche' else 'Automatische Segmentierung'}")
    
    # --- Bildvergleich mit Slider ---
    st.subheader("Bildvergleich")
    leafmap.image_comparison(
        results['tiff_path'],
        results['vis_path'],
        label1="Satellitenbild",
        label2="Segmentierung",
        starting_position=50,
        width=700
    )
    
    # --- Persistente Karte ---
    result_map = leafmap.Map(height=700)
    result_map.add_raster(results['tiff_path'], layer_name="Satellitenbild")
    
    # Farbpalette für Raster festlegen
    palette = "viridis" if results['process_type'] == "Automatische Segmentierung" else "Greens"
    
    result_map.add_raster(
        results['mask_path'], 
        layer_name="Segmentierung", 
        opacity=0.7,
        palette=palette,
        nodata=0
    )
    
    # Korrekte Vektorisierung
    if results['process_type'] == "Text-Prompt Suche":
        results['sam'].raster_to_vector(results['mask_path'], results['vector_path'])
    else:
        sam = SamGeo()
        sam.raster_to_vector(results['mask_path'], results['vector_path'])
    
    # Vektorlayer mit korrektem Stil
    result_map.add_vector(
        results['vector_path'], 
        layer_name="Vektorisierte Objekte",
        style={
            "color": "#3388ff",
            "weight": 2,
            "fillOpacity": 0.5
        }
    )
    
    result_map.add_layer_control()
    result_map.to_streamlit(key="result_map")

    # --- Korrekter Shapefile-Zip Export ---
    st.header("Ergebnisse exportieren")
    
    def create_shapefile_zip(shp_path, zip_path):
        """Erstellt ein korrektes Shapefile-Zip mit allen benötigten Dateien"""
        base = os.path.splitext(shp_path)[0]
        extensions = ['.shp', '.shx', '.dbf', '.prj']
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in extensions:
                file_path = base + ext
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))
    
    # Geodaten-Export
    cols = st.columns(3)
    
    # GeoTIFF Download
    cols[0].download_button(
        label="GeoTIFF herunterladen",
        data=open(results['mask_path'], "rb"),
        file_name="segmentation.tif"
    )
    
    # Visualisierung Download
    cols[1].download_button(
        label="Visualisierung herunterladen",
        data=open(results['vis_path'], "rb"),
        file_name="visualization.png"
    )
    
    # Shapefile Download
    zip_path = "segmentation_shp.zip"
    create_shapefile_zip(results['vector_path'], zip_path)
    
    cols[2].download_button(
        label="Vektordaten (Shapefile)",
        data=open(zip_path, "rb"),
        file_name="segmentation_shp.zip"
    )
    
    # Reset-Button
    if st.button("Neue Segmentierung starten"):
        st.session_state.results = None
        st.session_state.map_visible = True
        st.rerun()