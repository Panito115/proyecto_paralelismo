'''
@tortolala
Pokemon image processing pipeline.
'''

import requests
import time
import os
import threading
from queue import Queue
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


def pipeline_pokemon(
    n=150,
    dir_origin='pokemon_dataset',
    dir_name='pokemon_processed',
    download_workers=8,
    process_workers=None,
    queue_size=32
):
    '''
    Descarga y procesa imágenes en paralelo utilizando colas y pools.
    '''

    os.makedirs(dir_origin, exist_ok=True)
    os.makedirs(dir_name, exist_ok=True)

    indices = range(1, n + 1)
    total = len(indices)
    base_url = 'https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/imagesHQ'

    print(f'\nPipeline: descargando y procesando {total} imágenes...\n')

    if total == 0:
        print('  No se solicitaron imágenes.\n')
        return {
            'download_time': 0.0,
            'processing_time': 0.0,
            'total_time': 0.0,
            'processed': 0,
            'download_errors': [],
            'processing_errors': []
        }

    if process_workers is None:
        process_workers = os.cpu_count() or 1

    process_workers = max(1, min(process_workers, total))

    queue = Queue(maxsize=queue_size)
    download_errors = []
    processing_errors = []
    process_futures = []
    process_map = {}

    pipeline_start = time.time()
    download_start = time.time()
    processing_started_at = None
    sentinel = object()

    with ProcessPoolExecutor(max_workers=process_workers) as process_pool:

        def consumer(pool):
            nonlocal processing_started_at

            while True:
                image_name = queue.get()
                if image_name is sentinel:
                    queue.task_done()
                    break

                if processing_started_at is None:
                    processing_started_at = time.time()

                future = pool.submit(_process_image, (dir_origin, dir_name, image_name))
                process_futures.append(future)
                process_map[future] = image_name
                queue.task_done()

        consumer_thread = threading.Thread(target=consumer, args=(process_pool,), name='pokemon-consumer')
        consumer_thread.start()

        with ThreadPoolExecutor(max_workers=download_workers) as download_pool:
            download_futures = {
                download_pool.submit(_fetch_pokemon, base_url, dir_origin, index): index
                for index in indices
            }

            for future in tqdm(
                as_completed(download_futures),
                total=len(download_futures),
                desc='Descargando',
                unit='img'
            ):
                index = download_futures[future]
                try:
                    file_name = future.result()
                    queue.put(file_name)
                except Exception as e:
                    download_errors.append(f'{index:03d}.png: {e}')
                    tqdm.write(f'  Error descargando {index:03d}.png: {e}')

        download_time = time.time() - download_start

        queue.put(sentinel)
        queue.join()
        consumer_thread.join()

    processing_bar_total = len(process_futures)
    processing_bar = tqdm(total=processing_bar_total, desc='Procesando', unit='img')

    for future in as_completed(process_futures):
        image_name = process_map[future]
        try:
            future.result()
        except Exception as e:
            processing_errors.append(f'{image_name}: {e}')
            tqdm.write(f'  Error procesando {image_name}: {e}')
        finally:
            processing_bar.update(1)

    processing_bar.close()

    pipeline_end = time.time()
    processing_time = 0.0
    if processing_started_at is not None:
        processing_time = pipeline_end - processing_started_at

    total_time = pipeline_end - pipeline_start
    processed = processing_bar_total

    if processed:
        print(f'\n  Pipeline completado en {total_time:.2f} segundos')
        print(f'  Descarga efectiva: {download_time:.2f} s ({download_time/processed:.2f} s/img)')
        print(f'  Procesamiento efectivo: {processing_time:.2f} s ({processing_time/processed:.2f} s/img)\n')
    else:
        print('\n  No se procesaron imágenes.\n')

    return {
        'download_time': download_time,
        'processing_time': processing_time,
        'total_time': total_time,
        'processed': processed,
        'download_errors': download_errors,
        'processing_errors': processing_errors
    }


if __name__ == '__main__':

    print('='*60)
    print_pikachu()
    print('   POKEMON IMAGE PROCESSING PIPELINE')
    print('='*60)
    
    # Pipeline: Descarga y procesamiento en paralelo
    metrics = pipeline_pokemon()
    download_time = metrics['download_time']
    processing_time = metrics['processing_time']
    total_time = metrics['total_time']
    processed = metrics['processed']

    print('='*60)
    print('RESUMEN DE TIEMPOS\n')
    print(f'  Descarga:        {download_time:.2f} seg')
    print(f'  Procesamiento:   {processing_time:.2f} seg')
    print(f'  Imágenes:        {processed}')
    print(f'  Total pipeline:  {total_time:.2f} seg')
    print('='*60)
