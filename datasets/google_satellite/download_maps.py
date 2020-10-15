import numpy as np
import click, os.path as osp
import datasets.google_satellite.static_map as GoogleStaticMap

class MapDownloader:
    def __init__(self, store_dir, gmap, storage_format="png", location_spec_fname="lat_lng_spec.txt"):
        self.store_dir = store_dir
        self.location_spec_fname = location_spec_fname
        self.storage_format = storage_format
        self.gmap = gmap

    def download(self):
        location_spec_file = osp.join(self.store_dir, self.location_spec_fname)
        lat_lst, lng_lst = [], []
        with open(location_spec_file) as f:
            for line in f.readlines():
                lat, lng = line.strip().split(", ")
                lat, lng = float(lat), float(lng)
                lat_lst.append(lat)
                lng_lst.append(lng)
        self.gmap.store_many(lat_lst, lng_lst, self.store_dir, fname_format="prefix_index", prefix="map",
                        storage_format=self.storage_format)

@click.command()
@click.option('-i', '--store-dir',  required=True, type=str, help="Directory with lat_lng_list.txt.")
@click.option('-k', '--api-key-dir',  required=False, default="./datasets/google_satellite/",
              type=str, help="Directory with google maps API key.")
@click.option('-z', '--zoom', default=10, required=False, type=int, help='Zoom level.')
@click.option('-s', '--size', default="640x640", required=False, type=str, help='Image size.')
# @click.option('-f', '--file-name-format', default="prefix_index", required=False, type=str, help='Image size.')
@click.option('-t', '--map-type', default="satellite", required=False, type=str,
              help='Map type (roadmap, satellite, terrain, or hybrid).')
@click.option('-m', '--mode', default="RGB", required=False, type=str,
              help='File mode: RGB (default) or Gray.')
@click.option('--storage-fmt', default="png", required=False, type=str,
              help='Map storage format (default: png).')
def download_maps(store_dir, api_key_dir, zoom, size, map_type, mode, storage_fmt):
    # print(store_dir, api_key_dir, zoom, size, map_type, mode, storage_fmt)
    api_key = GoogleStaticMap.APIFile(osp.join(api_key_dir, "api_key.npy"))
    gmap = GoogleStaticMap.GoogleStaticMap(api_key, zoom=zoom, size=size, maptype=map_type, mode=mode, verbose=True)
    md = MapDownloader(store_dir, gmap, storage_fmt)
    md.download()

if __name__ == "__main__":
    download_maps()