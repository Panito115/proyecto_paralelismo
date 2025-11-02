'''
@tortolala
Pokemon image processing pipeline.
'''

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from pika_banner import print_pikachu
from tqdm import tqdm
import requests
import time
import os


def download_pokemon(n=150, dir_name='pokemon_dataset'):
    '''
    Descarga las imágenes de los primeros n Pokemones.
    '''

    os.makedirs(dir_name, exist_ok=True)
    base_url = 'https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/imagesHQ' 

    print(f'\nDescargando {n} pokemones...\n')
    start_time = time.time()
    
    for i in tqdm(range(1, n + 1), desc='Descargando', unit='img'):
        file_name = f'{i:03d}.png'
        url = f'{base_url}/{file_name}'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            img_path = os.path.join(dir_name, file_name)
            with open(img_path, 'wb') as f:
                f.write(response.content)
                
        except requests.exceptions.RequestException as e:
            tqdm.write(f'  Error descargando {file_name}: {e}')
    
    total_time = time.time() - start_time
    print(f'  Descarga completada en {total_time:.2f} segundos')
    print(f'  Promedio: {total_time/n:.2f} s/img')
    
    return total_time


def process_pokemon(dir_origin='pokemon_dataset', dir_name='pokemon_processed'):
    '''
    Procesa las imágenes aplicando múltiples transformaciones.
    '''

    os.makedirs(dir_name, exist_ok=True)
    images = sorted([f for f in os.listdir(dir_origin) if f.endswith('.png')])
    total = len(images)
    
    print(f'\nProcesando {total} imágenes...\n')
    start_time = time.time()
    
    for image in tqdm(images, desc='Procesando', unit='img'):
        try:
            path_origin = os.path.join(dir_origin, image)
            img = Image.open(path_origin).convert('RGB')
            
            # Transformaciones a imagen (CPU-intensive task)
            img = img.filter(ImageFilter.GaussianBlur(radius=10))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img_inv = ImageOps.invert(img)
            img_inv = img_inv.filter(ImageFilter.GaussianBlur(radius=5))
            width, height = img_inv.size
            img_inv = img_inv.resize((width * 2, height * 2), Image.LANCZOS)
            img_inv = img_inv.resize((width, height), Image.LANCZOS)
        
            saving_path = os.path.join(dir_name, image)
            img_inv.save(saving_path, quality=95)
            
        except Exception as e:
            tqdm.write(f'  Error procesando {image}: {e}')
    
    total_time = time.time() - start_time
    print(f'  Procesamiento completado en {total_time:.2f} segundos')
    print(f'  Promedio: {total_time/total:.2f} s/img\n')
    
    return total_time


if __name__ == '__main__':

    print('='*60)
    print_pikachu()
    print('   POKEMON IMAGE PROCESSING PIPELINE')
    print('='*60)
    
    # Fase 1: Descarga (I/O Bound)
    download_time = download_pokemon()
    
    # Fase 2: Procesamiento (CPU Bound)
    processing_time = process_pokemon()
    
    # Resumen final
    total_time = download_time + processing_time

    print('='*60)
    print('RESUMEN DE TIEMPOS\n')
    print(f'  Descarga:        {download_time:.2f} seg')
    print(f'  Procesamiento:   {processing_time:.2f} seg\n')
    print(f'  Total:           {total_time:.2f} seg')
    print('='*60)