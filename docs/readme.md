# Come usare lo script...
1) Serve che ci sia la cartella data/ nella stessa directory del .py
2) Serve che ci sia la cartella media/ nella stessa directory del .py
3) Serve avere un python .venv o un altro virtual env o un altro python con le seguenti dipendenze:
    - numpy
    - seaborn
    - matplotlib
    - pandas
    (Chiedere a Matteo se non si sa cosa sia)

4) per lanciare lo script python si usa il seguente comando:
    - python3 main_temporal_query_data.py --query_name \<nome_csv_file\>

5) se si vuole creare il singolo alert con una query che mostri un singolo alert bisogna aggiungere nel comando anche il seguente:
    - python3 main_temporal_query_data.py --query_name \<nome_csv_file\> --do_single_alert

6) se non si vuole la tabella, ma solo il temporal, aggiungere nel comando anche il seguente:
    - python3 main_temporal_query_data.py --query_name \<nome_csv_file\> --not_do_table