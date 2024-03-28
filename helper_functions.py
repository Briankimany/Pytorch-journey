
import torch
import matplotlib.pyplot as plt 
import math , random
import subprocess
from copy import deepcopy

from typing import Dict , List
from torch import nn


from sklearn.metrics import classification_report
from timeit import default_timer as timer

def install_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Installing...")
        
        subprocess.check_call(["pip", "install", package_name,'-q'])
        print(f"Package '{package_name}' has been installed successfully.")
    else:
        print(f"Package '{package_name}' is already installed.")

from tqdm.auto import tqdm
try:
    from torchmetrics.classification import MulticlassAccuracy , MulticlassAUROC  , MulticlassPrecision , MulticlassF1Score , MulticlassRecall , MulticlassConfusionMatrix
except Exception as e:
    print("installing torchmetrics")
    install_package("torchmetrics")
    from torchmetrics.classification import MulticlassAccuracy , MulticlassAUROC  , MulticlassPrecision , MulticlassF1Score , MulticlassConfusionMatrix , MulticlassRecall


def plot(data ,  reverse_mapping_dict):
  """
    Plot images from the dataset.

    Args:
        data (list): List of tuples, each containing an image tensor and its corresponding label.
        reverse_mapping_dict (dict): Dictionary mapping label indices to their corresponding class names.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure containing the plotted images.
  """
  
  cols = round(math.sqrt(len(data)))
  rows = math.ceil(len(data) / cols)

  cols_ = max(rows , cols)
  rows_ = min(rows , cols)
  print(f"Cols {cols_}  Rows {rows_} , {len(data)}")
  fig , axs = plt.subplots(rows_ , cols_ , figsize = (10 , 5))
  fig.subplots_adjust(wspace=0.4, hspace=0.4)

  taken = []

  for row in range( rows_ ):
    for col in range( cols_ ):
      img_idx = row * cols_ + col
      if img_idx < len(data):
        image = data[img_idx][0]
        image = image.squeeze()
        img_key = int(data[img_idx][1])
        label = reverse_mapping_dict[img_key]

        if image.shape[0] == 3:
          image = image.permute(1 , 2 , 0)

        axs[row , col].imshow( image, cmap = 'gray')
        axs[row , col].axis("off")
        axs[row , col].set_title(label)
      else:
        axs[row , col].axis("off")


  return fig


def save_model(model:torch.nn.Module , path):
  torch.save(model.state_dict() , f = path)


def train_step(model:nn.Module ,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module ,
               optimizer : torch.optim.Optimizer,
               fine_tune= None,
               device = None
              ):
    
  """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.

    Returns:
        average_loss (float): Average loss over the training data for this step. 
        average_acc (float or None): Average accuracy over the training data for this step, if accuracy_fn is provided.
                                      None otherwise.


   """

  total_correct , total_loss , total_samples= 0 , 0 , 0
  model.train()
  
  if device == None:
    device = "cuda"  if torch.cuda.is_available() else "cpu"
    
  if fine_tune:
    fine_tune = fine_tune.to(device)
    
  for batch , (X , y)  in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)

    if fine_tune:
      with torch.no_grad():
        X = fine_tune(X)
      
    y_logits = model(X)

    loss = loss_fn(y_logits , y)
    total_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(y_logits, dim=1)
    total_correct += (y_pred_class == y).sum().item()
    total_samples += y.size(0)

  average_loss = total_loss / len(dataloader)
  average_acc = total_correct / total_samples
  
  return average_loss , average_acc



def test_step(model: nn.Module,           
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=None,
              fine_tune = None,
              ):
    
    """
    Perform a single evaluation step.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        loss_fn (torch.nn.Module): Loss function.
        device (str, optional): Device to use for evaluation ('cpu' or 'cuda'). If None, the function will
                                automatically select the appropriate device based on GPU availability. Default is None.

    Returns:
        average_loss (float): Average loss over the evaluation data for this step.
        test_accuracy (float): Accuracy over the evaluation data for this step.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            if fine_tune:
              X = fine_tune(X)
              
            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            total_loss += loss.item()

            y_pred_class = torch.argmax(y_logits, dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += y.size(0)

    average_loss = total_loss / len(dataloader)
    test_accuracy = total_correct / total_samples
    # print(f"Test Loss: {average_loss:.6f} | Test Accuracy: {test_accuracy:.6f}")

    return average_loss, test_accuracy

def train(model:torch.nn.Module ,
          epochs:int,
          train_dataloader :torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optim: torch.optim.Optimizer,
          loss_fn :torch.nn.Module,
          device = None,
          fine_tune = None,
          call_backs = False):

  history = {
      "train_loss":[],
      "train_acc":[],
      'test_acc':[],
      "test_loss":[]
      }


  for epoch in tqdm(range(1 ,epochs+1)):
    train_loss , train_acc = train_step(model = model,
                                        dataloader = train_dataloader,
                                        loss_fn = loss_fn,
                                        optimizer = optim,
                                        device = device,
                                        fine_tune = fine_tune
                                        
                                        )

    test_loss , test_acc = test_step(model = model,
                                    dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    device = device,
                                    fine_tune= fine_tune
                                    )

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    
    
    if call_backs:
      if min(history['test_loss']) == history['test_loss'][-1]:
        best_model = deepcopy(model)
        save_model(best_model, 'best_model.pt')
      

    # print("===="* 70)
    info = "{} : Train Loss {:.3f} | Train Accuracy {:.3f} || Test Loss {:.3f} | Test Accuracy {:.3f}".format(epoch ,train_loss , train_acc , test_loss , test_acc)
    print(info)

  return history


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

def eval_model(model,
               dataloader : torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               num_classes,
               model_name = None ,
               full_report = False,
               device = None):


  """
    Evaluate the performance of a PyTorch model on a given dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation dataset.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        num_classes (int): Number of classes in the classification task.
        model_name (str, optional): Name of the model. Default is None.
        full_report (bool, optional): Whether to return a detailed report for each batch. Default is False.
        device (str, optional): Device to use for evaluation ('cpu' or 'cuda'). If None, the function will
                                automatically select the appropriate device based on GPU availability. Default is None.

    Returns:
        tuple or dict: Evaluation results. If full_report is False, returns a dictionary with average metrics.
                       If full_report is True, returns a tuple containing a dictionary with batch-wise metrics and
                       a dictionary with average metrics.
   """

  if device == None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  model.to(device)
  model_eval = {
      "NAME":model_name,
      "auroc":[],
      "report":[],
      "Matrix":[],
      "accuracy":[],
      "loss":[],
      "f1_score":[],
      "recall":[],
      "precision":[],
      "inference_time":[]
  }


  accuracy_fn = MulticlassAccuracy(num_classes = num_classes).to(device)

  with torch.inference_mode():
    accuracy , loss = 0 , 0

    for (X_test , y_test)  in tqdm(dataloader):

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        start = timer()
        test_pred = model(X_test)

        end = timer()
        duration = end - start

        pred_probs = torch.softmax(test_pred , dim = 1)

        current_loss =  loss_fn(test_pred , y_test)
        loss +=current_loss

        current_accuracy = accuracy_fn(test_pred.argmax(dim = 1),y_test)
        accuracy += current_accuracy

        auroc = MulticlassAUROC(num_classes)
        Precision = MulticlassPrecision(num_classes)
        Recall = MulticlassRecall(num_classes)
        F1_score  = MulticlassF1Score(num_classes)


        ### Transfering the pred probs to cpu for evaluation
        if device == 'cuda':
          pred_probs = pred_probs.to('cpu')
          y_test = y_test.to("cpu")
          test_pred = test_pred.to("cpu")

        auroc_score  = auroc( torch.softmax(test_pred , dim = 1) , y_test)
        f1_score = F1_score(pred_probs , y_test)
        recall_score = Recall(pred_probs , y_test )
        precision_score = Precision(pred_probs , y_test)

        report = classification_report(y_test,test_pred.argmax(dim = 1))

        conf_matrix = MulticlassConfusionMatrix(num_classes)
        conf_m = conf_matrix(pred_probs , y_test)

        model_eval["auroc"].append(auroc_score)
        model_eval['precision'].append(precision_score)
        model_eval['recall'].append(recall_score)
        model_eval['f1_score'].append(f1_score)

        model_eval["report"].append(report)
        model_eval["Matrix"].append(conf_matrix)

        model_eval["accuracy"].append(current_accuracy)
        model_eval["loss"].append(current_loss)
        model_eval['inference_time'].append(duration)


    loss = float(loss/len(dataloader))
    accuracy = float(accuracy /len(dataloader))

    model_eval["AV_ACC"] = accuracy
    model_eval["AV_LOSS"] = loss

    average_recall = float(torch.tensor(model_eval['recall']).mean())
    average_precision = float(torch.tensor(model_eval['precision']).mean())
    average_f1 = float(torch.tensor(model_eval['f1_score']).mean())
    average_auroc = float(torch.tensor(model_eval['auroc']).mean())
    total_time = sum(model_eval['inference_time'])
    time_per_batch = float(torch.tensor(model_eval['inference_time']).mean())

    average_report = {
        "MODEL_NAME":model_name,
        "ACCURACY":accuracy,
        "LOSS":loss,
        "AUROC":average_auroc,
        "RECALL":average_recall,
        "F1_SCORE":average_f1,
        "PRECISION":average_precision,
        "TOTAL_TIME":total_time,
        "BATCH_TIME":time_per_batch
      }

  if full_report:
    return  model_eval , average_report
  return None  , average_report


if __name__ == "__main__":
  import os 
  print (os.getcwd())