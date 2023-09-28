!pip install pyspark==3.1.2
!pip install graphframes==0.8.2

!pip install findspark

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame

# Create a Spark session
spark = SparkSession.builder.appName("FlightAnalysis").getOrCreate()
!mv spark-hive_2.12-3.4.0.jar /usr/local/lib/python3.10/dist-packages/pyspark/jars/

# Read the flight data from CSV into a DataFrame
flight_data = spark.read.csv("flight_data.csv", header=True, inferSchema=True)

# Rename the columns for consistency
flight_data = flight_data.withColumnRenamed("airport_id", "id") \
                         .withColumnRenamed("airport_name", "name")

# Create vertices DataFrame
vertices = flight_data.select("id", "name").distinct()

# Create edges DataFrame
edges = flight_data.select("origin_city", "destination_city") \
                  .withColumnRenamed("origin_city", "src") \
                  .withColumnRenamed("destination_city", "dst")

# Create a GraphFrame from vertices and edges DataFrames
graph = GraphFrame(vertices, edges)

# Perform the necessary queries and computations

# Query 1 - Find the number of flights departing from each airport
outgoing_flights = graph.outDegrees
outgoing_flights.show()

# Query 2 - Find the number of flights arriving at each airport
incoming_flights = graph.inDegrees
incoming_flights.show()

# Query 3 - Find the number of flights between each pair of airports
flight_counts = graph.edges.groupBy("src", "dst").count()
flight_counts.show()

# Query 4 - Sort and print the longest routes
longest_routes = flight_counts.orderBy(col("count").desc())
longest_routes.show()

# Query 5 - Display highest degree vertices for incoming and outgoing flights of airports
highest_outgoing_degrees = outgoing_flights.orderBy(col("outDegree").desc())
highest_incoming_degrees = incoming_flights.orderBy(col("inDegree").desc())
highest_outgoing_degrees.show()
highest_incoming_degrees.show()

# Query 6 - Get the airport name with IDs 10397 and 12478
airport_names = vertices.filter(vertices.id.isin(10397, 12478))
airport_names.show()

# Query 7 - Find the airport with the highest incoming flights
airport_highest_incoming = incoming_flights.orderBy(col("inDegree").desc()).first()
print("Airport with the highest incoming flights:", airport_highest_incoming)

# Query 8 - Find the airport with the highest outgoing flights
airport_highest_outgoing = outgoing_flights.orderBy(col("outDegree").desc()).first()
print("Airport with the highest outgoing flights:", airport_highest_outgoing)

# Query 9 - Find the most important airports according to PageRank
page_rank = graph.pageRank(resetProbability=0.15, tol=0.01)
important_airports = page_rank.vertices.orderBy(col("pagerank").desc())
important_airports.show()

# Query 10 - Sort the airports by ranking
sorted_airports = important_airports.orderBy(col("pagerank"))
sorted_airports.show()

# Query 11 - Display the most important airports
top_airports = important_airports.limit(5)
top_airports.show()

# Query 12 - Find the routes with the lowest flight costs
lowest_costs = graph.edges.groupBy("src", "dst").agg({"src": "min"})
lowest_costs.show()

# Query 13 - Find airports and their lowest flight costs
airport_costs = lowest_costs.join(vertices, lowest_costs.src == vertices.id)
airport_costs.show()

# Query 14 - Find the average cost of flights for each airport
average_costs = graph.edges.groupBy("src", "dst").avg("src")
average_costs.show()

# Query 15 - Find the airports with the highest average flight costs
highest_average_costs = average_costs.orderBy(col("avg(src)").desc())
highest_average_costs.show()

# Query 16 - Find the airports with the lowest average flight costs
lowest_average_costs = average_costs.orderBy(col("avg(src)").asc())
lowest_average_costs.show()

# Query 17 - Find the shortest paths between airports
shortest_paths = graph.shortestPaths(landmarks=["10397", "12478"])
shortest_paths.show()

# Query 18 - Find the strongly connected components in the graph
strongly_connected_components = graph.stronglyConnectedComponents()
strongly_connected_components.show()

# Query 19 - Find the connected components in the graph
connected_components = graph.connectedComponents()
connected_components.show()

# Query 20 - Triangle count in the graph
triangle_count = graph.triangleCount()
triangle_count.show()

# Stop the Spark session
spark.stop()
