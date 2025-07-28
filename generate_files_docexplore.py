import os

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from loaders.dataset_docexplore_eval import DocExploreEval, DocExploreQueries
from models.vision_transformer import vit_base
from models.vit_lora import vit_lora

from tqdm import tqdm


if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ver si se esta aplicando padding a las imágenes
    image_dataset = DocExploreEval("/home/data/cstears/dataframes/dets_thr-01_df.pkl", transform=dataset_transforms)
    cataloges_dataset = DocExploreQueries("/home/mpizarro/iDoc/loaders/docexplore_photos_list.txt", transform=dataset_transforms)

    image_loader = DataLoader(dataset=image_dataset, batch_size=64, num_workers=12, shuffle=False)
    cataloges_loader = DataLoader(dataset=cataloges_dataset, batch_size=64, num_workers=12, shuffle=False)

    path_results = 'results/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    model = vit_lora().to(device)
    #vit_base().to(device) 
    #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
    
    model_checkpoint = torch.load("/home/mpizarro/out/mix_ibot_lora/teacher.pth", map_location=device, weights_only=False)
    model.load_state_dict(model_checkpoint['state_dict'])                
    model.eval()                                                         
    
    distance_fn = lambda x, y: F.cosine_similarity(x, y, dim=-1)

    catalog_embeddings = []
    catalog_labels = []
    catalog_paths = []
    top_k = len(image_dataset)
    archivo_salida = path_results + "lora"

    with torch.no_grad():
        for (catalog, s_label, path) in tqdm(cataloges_loader):
            catalog_feat = model(catalog.to(device))
            catalog_embeddings.append(catalog_feat) # [batch_size, 512]
            catalog_labels.append(s_label) # [batch_size]
            catalog_paths.append(path)

        catalog_embeddings = torch.cat(catalog_embeddings, dim=0) # [batch_size, 512] -> [n_catalog, 512]
        catalog_labels = torch.cat(catalog_labels, dim=0) # [batch_size] -> [n_catalog]
        catalog_paths = [path for paths in catalog_paths for path in paths]

    distance_values = []
    labels_images = []
    images_path = []
    bbox_images = []

    with torch.no_grad():
        print("Generando rankings")
        for (image, labe_bbox, path) in tqdm(image_loader):
            image_feat = model(image.to(device)) # [batch_size, 512]

            bbox = [l.split("_")[-1].split("-") for l in labe_bbox]
            bbox = [[int(c) for c in b] for b in bbox]
            label = torch.tensor([int(l.split("_")[0]) for l in labe_bbox])
            
            distance = distance_fn(catalog_embeddings.unsqueeze(1), image_feat.unsqueeze(0))

            distance_values.append(distance)
            labels_images.append(label.unsqueeze(0))
            bbox_images.append(torch.tensor(bbox).unsqueeze(0))
            images_path.append(path)

        all_images_path = [path for paths in images_path for path in paths]
        
        all_query_distance = torch.cat(distance_values, dim=1).to(device) # [queries, total_images]
        all_labels_images = torch.cat(labels_images, dim=1).to(device) # [1, total_images]
        all_bbox_images = torch.cat(bbox_images, dim=1).to(device) # [1, total_images]

        torch.cuda.empty_cache()
        del distance_values
        del labels_images
        del images_path
        del bbox_images

        max_values, max_indices = torch.topk(all_query_distance, 2000, dim=1, largest=True, sorted=True) # [n_cataloges, top_k]
        max_labels = all_labels_images[0][max_indices]
        max_bbox = all_bbox_images[0][max_indices]

        top_images_path = []
        for i in range(len(max_indices)):
            tmp = []
            for j in range(10):
                tmp.append(all_images_path[max_indices[i][j]])
            top_images_path.append(tmp)

        for size in [1000]:
            print(f"Guardando los resultados de los primeros {size} rankings")
            with open(archivo_salida + "_" + str(size) + ".txt", "w") as f:
                for i in range(len(catalog_labels)):
                    max_label_row = max_labels[i][all_query_distance[i][max_indices[i]] >= 0][:size]
                    max_bbox_row = max_bbox[i][all_query_distance[i][max_indices[i]] >= 0][:size]

                    path_image = catalog_paths[i].split("/")[-1].split(".")[0]

                    final_str = ""
                    for label, bbox in zip(max_label_row, max_bbox_row):
                        max_label_str = str(label.item())
                        max_bbox_str = "-".join(map(str, bbox.tolist()))
                        final_str += f"{max_label_str}-{max_bbox_str} "
                    
                    f.write(f"{path_image}:{final_str}\r\n")
            
            print(f"Guardando los scores de los primeros {size} rankings")
            with open(archivo_salida + "_scores.txt", "w") as f:
                for i in range(len(catalog_labels)):
                    max_values_row = max_values[i]
                    path_image = catalog_paths[i].split("/")[-1].split(".")[0]

                    final_str = ""
                    for max_value in max_values_row:
                        max_value_str = str(max_value.item())
                        final_str += f"{max_value_str[:5]}"
                    
                    f.write(f"{path_image}:{final_str}\r\n")
                

        with open(archivo_salida + "_images.txt", "w") as f:
            for i in range(len(catalog_paths)):
                catalog_label = catalog_labels[i]
                catalog_path = catalog_paths[i]
                top_images_path_row = top_images_path[i]
                catalog_label_str = str(catalog_path)
                top_images_path_row_str = ", ".join(map(str, top_images_path_row))
                f.write(f"{catalog_label}, {catalog_label_str}, {top_images_path_row_str}\n")

        print("Valores guardados en", archivo_salida)
    