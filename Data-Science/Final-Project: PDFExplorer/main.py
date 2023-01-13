from tabnanny import verbose
from pdfexplorer import PDFExplorer
import os 
import time 

if __name__ == "__main__":

    #The User needs to specify the path to the pdf files 
    print("\nWelcome to the PDFExplorer!. With this little tool you can visualize and organize your PDF files easily.\n")
    print("First you need to specify the relative path to the PDF files:")
    path = input()
    while not os.path.exists(path):
        print("The directory you specified does not exist. Please try again")
        path = input()
    os.system("clear")
    print("\nPDFExplorer is loading and preprocessing the PDF files!. Do you want verbose output? (y/n)")
    verbose = input()
    if verbose == "y":
        verbose_output = True 
    elif verbose == "n":
        verbose_output = False 
    else: 
        print("Invalid input. Defaults to non-verbose output")
        time.sleep(2)
        verbose_output = False
    os.system("clear")
    #Setting up an instance of the PDFExplorer, loading the pdf files into memory, nlp-preprocessing, 
    #creating a bag-of-words-matrix and finally calculating the similarity between the texts
    explorer = PDFExplorer(path)
    explorer.load_pdfs(verbose=verbose_output)
    print("\nPreprocessing the pdf files (this could take some time)")
    explorer.preprocess(verbose=verbose_output)
    print("\nPreprocessing done!")
    print("Setting all relevant attributes ...")
    explorer.create_bow_matrix()
    explorer.calculate_cos_sim()
    print("Entering the main loop of the program")
    print("If you need help, just enter 'help' to see all the commands")
    time.sleep(5)
    os.system("clear")
    #Entering the main loop 
    run_programm = True 
    while run_programm:
        user_input = input()
        if user_input == "help": 
            print("""Here is a list of the available commands:\n
            visualize - visualizes the similarity of the PDF files!\n
            \t Tipp: You can visualize the PDF files before clustering to get an idea of the appropriate number of clusters\n
            cluster - groups similiar PDF files together\n 
            \t You can either specify a number of clusters or let the PDFExplorer decide!\n
            organize - organizes the directory by creating folders for each cluster and either copying or moving the resp. files\n
            similar - displays a list of all the PDF files that are similar to a specified PDF file\n
            clear - clears the terminal output\n
            exit - exits the PDFExplorer
            """)
        elif user_input == "visualize":
            explorer.visualize_articles()
            os.system("clear")
        elif user_input == "cluster": 
            print("Specify the number of clusters. Type 'automatic' if you want the PDFExplorer to decide")
            cluster_input = input()
            if cluster_input == "automatic": 
                explorer.cluster_articles(verbose=True)
                time.sleep(2)
                os.system("clear")
            else:
                try:  
                    number_of_clusters = int(cluster_input)
                    explorer.cluster_articles(number_of_clusters)
                    os.system("clear")
                except ValueError: 
                    print("Input has to be a number. Try again!")
                    print("Returning to main loop..")
                    time.sleep(2)
                    os.system("clear")
                    continue 
        elif user_input == "organize": 
            if not explorer.clustering_done: 
                print("You have to cluster first!")
                print("Returning to main loop")
                time.sleep(2)
                os.system("clear")
                continue
            print("Do you want to copy or move the files? (c/m)")
            copy_move_input = input()
            if copy_move_input == "c": 
                explorer.organize_articles()
            elif copy_move_input == "m":
                explorer.organize_articles(copy=False)
            else: 
                print("Invalid command! Type in 'c' for copying or 'm' for moving!") 
                print("Returning to main loop!")
                time.sleep(2)
                os.system("clear")
                continue
            os.system("clear")
        
        elif user_input == "similar":
            if not explorer.clustering_done: 
                print("You have to cluster first!")
                print("Returning to main loop")
                time.sleep(2)
                os.system("clear")
                continue
            print("Specify the file's name:")
            article_name = input()
            if article_name not in explorer.aliases.values(): 
                print("Could not find the pdf file!")
                print("Returning to main loop")
                time.sleep(2)
                os.system("clear")
                continue 
            similar_articles = explorer.get_similar_articles(article_name)
            for article in similar_articles:
                print("\n")
                print(article)
                print("\n")

        elif user_input == "exit":
            run_programm = False
        elif user_input == "clear":
            os.system("clear")
        else: 
            print("The command you entered is not valid!")
            print("Returning to main loop!")
            time.sleep(2)
            os.system("clear")

        