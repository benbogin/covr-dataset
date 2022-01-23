Here you will find the code necessary to generate the COVR dataset.

Note that that the code that uses LXMERT to verify distracting images is not included yet, thus
generated questions might often be noisy. Please contact us for info on how to use with LXMERT!
### Setting up environment
Install all required packages
```
pip install -r requirements.txt
```

### Setting up neo4j
First, you will need to set up neo4j. This is the graph database that we query to get relevant scene graphs.
   
1. Install neo4j community edition (free):
   https://neo4j.com/download-center/#community

   We used version 4.2.1. Most likely, any 4.2.* version is compatible.

   ```
   wget https://neo4j.com/artifact.php?name=neo4j-community-4.2.1-unix.tar.gz -O neo4j.tar.gz
   tar -xvf neo4j.tar.gz
   ```
2. We want to create the graph database such that it will contain all GQA and imsitu scene graphs.
   The following script will download GQA and imsitu, and create the necessary files for neo4j to import:
   ```
   python scripts/index_neo4j.py --neo4j-import-path ../../../neo4j-community-4.2.1/import/
   ```
   
   You might need to change `--neo4j-import-path` according to your neo4j installation path.
   
   Make sure the `neo4j-community-4.2.1/import` directory contains both `relations.csv` and `objects.csv` files.
3. Import neo4j files:
   ```
   cd ../neo4j-community-4.2.1
   ./bin/neo4j-admin import --nodes import/objects.csv --relationships import/relations.csv
   ```
   
   You may need to install appropiate Java version. At the end of the import script, you will see:
   ```
   IMPORT DONE in 16s 801ms.
   Imported:
   1825127 nodes
   2259275 relationships
   7731210 properties
   ```
4. Disable authentication (or set it up if needed):
   1. Open `neo4j-community-4.2.1/conf/neo4j.conf`
   2. Uncomment `dbms.security.auth_enabled=false`
   3. Uncomment `dbms.default_listen_address=0.0.0.0`
5. Finally, run `neo4j`:
   ```
   ./bin/neo4j console
   ```

### Setting up redis
Redis is used to cache results of queries. This is not significant if the dataset is generated a single-time, however multiple executions
of the script will be much faster with caching.
1. Install redis
   ```
   sudo apt-get install redis
   ```

   (optionally with conda: `conda install -c anaconda redis`)
2. Run redis server
   ```
   redis-server
   ```

### Generate dataset
Use the following command to generate the dataset:
```
python generate.py --multiproc 16 --split train
python generate.py --multiproc 16 --split val
```

With `--multiproc` as the number of processes to be used.