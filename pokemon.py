'''
@tortolala
Pokemon image processing pipeline.
'''

import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from pika_banner import print_pikachu
from tqdm import tqdm


def _fetch_pokemon(base_url, dir_name, index, timeout=10):
    '''
    Descarga una imagen individual de Pokémon.
    '''

    file_name = f'{index:03d}.png'
    url = f'{base_url}/{file_name}'

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    img_path = os.path.join(dir_name, file_name)
    with open(img_path, 'wb') as f:
        f.write(response.content)

    return file_name


def download_pokemon(n=150, dir_name='pokemon_dataset', workers=8):
    '''
    Descarga las imágenes de los primeros n Pokemones.
    '''

    os.makedirs(dir_name, exist_ok=True)
    base_url = 'https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/imagesHQ' 

    print(f'\nDescargando {n} pokemones...\n')
    start_time = time.time()
    indices = range(1, n + 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_pokemon, base_url, dir_name, index): index
            for index in indices
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Descargando',
            unit='img'
        ):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                tqdm.write(f'  Error descargando {idx:03d}.png: {e}')
    
    total_time = time.time() - start_time
    print(f'  Descarga completada en {total_time:.2f} segundos')
    print(f'  Promedio: {total_time/n:.2f} s/img')
    
    return total_time


def _process_image(task):
    '''
    Aplica las transformaciones de imagen para un Pokémon.
    '''

    dir_origin, dir_name, image = task

    path_origin = os.path.join(dir_origin, image)

    with Image.open(path_origin) as img:
        img = img.convert('RGB')

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

    return image


def process_pokemon(dir_origin='pokemon_dataset', dir_name='pokemon_processed', workers=None):
    '''
    Procesa las imágenes aplicando múltiples transformaciones.
    '''

    os.makedirs(dir_name, exist_ok=True)
    images = sorted([f for f in os.listdir(dir_origin) if f.endswith('.png')])
    total = len(images)
    
    print(f'\nProcesando {total} imágenes...\n')
    start_time = time.time()

    if total == 0:
        print('  No se encontraron imágenes para procesar.\n')
        return 0.0

    if workers is None:
        workers = os.cpu_count() or 1

    max_workers = max(1, min(workers, total))

    tasks = [(dir_origin, dir_name, image) for image in images]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_image, task): task[2] for task in tasks}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Procesando',
            unit='img'
        ):
            try:
                future.result()
            except Exception as e:
                image_name = futures[future]
                tqdm.write(f'  Error procesando {image_name}: {e}')

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
