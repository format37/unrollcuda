# Find the docker container with name cuda-tips
container_id=$(sudo docker ps | grep cuda-tips | awk '{print $1}')
echo "container id: $container_id"
# Stop the container
sudo docker stop $container_id