#!/bin/sh

# ec2_run.sh
# Dave Gottlieb 2016
#
# Spin up EC2 spot instance and run model with desired parameters, keeping checkpoints in case of instance loss. 
# Based on gaming script from that one guy. Uses AWS CLI tools and jq. 

set -e 

export AWS_DEFAULT_REGION=us-west-1
export EC2_SECURITY_GROUP_ID=sg-a4647bc1

# Get the current lowest price for the GPU machine we want
echo -n "Getting lowest g2.2xlarge bid... "
PRICE=$( aws ec2 describe-spot-price-history --instance-types g2.2xlarge --product-descriptions "Linux/UNIX" --start-time `date +%s` | jq --raw-output '.SpotPriceHistory[].SpotPrice' | sort | head -1 )
echo $PRICE

echo -n "Looking for the theano AMI... "
AMI_SEARCH=$( aws ec2 describe-images --owner self --filters Name=name,Values=theano_machine )
if [ $( echo "$AMI_SEARCH" | jq '.Images | length' ) -eq "0" ]; then
	echo "not found. You must use gaming-down.sh after your machine is in a good state."
	exit 1
fi
AMI_ID=$( echo $AMI_SEARCH | jq --raw-output '.Images[0].ImageId' )
echo $AMI_ID

# If price is less than on-demand price minus $0.1, make a spot request
if echo "$PRICE < 0.602" | bc | -gt 0; then
	echo -n "Creating spot instance request... "
	SPOT_INSTANCE_ID=$( aws ec2 request-spot-instances --spot-price $( bc <<< "$PRICE + 0.05" ) --launch-specification "
	  {
	    \"SecurityGroupIds\": [\"$EC2_SECURITY_GROUP_ID\"],
	    \"ImageId\": \"$AMI_ID\",
	    \"InstanceType\": \"g2.2xlarge\",
	    \"KeyName\": \"th testbed\"
	  }" | jq --raw-output '.SpotInstanceRequests[0].SpotInstanceRequestId' )
	echo $SPOT_INSTANCE_ID

	echo -n "Waiting for instance to be launched... "
	aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids "$SPOT_INSTANCE_ID"
	
	INSTANCE_ID=$( aws ec2 describe-spot-instance-requests --spot-instance-request-ids "$SPOT_INSTANCE_ID" | jq --raw-output '.SpotInstanceRequests[0].InstanceId' )
	echo "$INSTANCE_ID"

	echo "Removing the spot instance request..."
	aws ec2 cancel-spot-instance-requests --spot-instance-request-ids "$SPOT_INSTANCE_ID" > /dev/null

	echo -n "Getting ip address... "
	IP=$( aws ec2 describe-instances --instance-ids "$INSTANCE_ID" | jq --raw-output '.Reservations[0].Instances[0].PublicIpAddress' )
	echo "$IP"

else
	echo -n "Spot price too high, creating on-demand instance." 

	INSTANCE_ID=$( aws ec2 run-instances --associate-public-ip-address --cli-input-json "
	  {
	    \"SecurityGroupIds\": [\"$EC2_SECURITY_GROUP_ID\"],
	    \"ImageId\": \"$AMI_ID\",
	    \"InstanceType\": \"g2.2xlarge\",
	    \"KeyName\": \"th testbed\"
	  }" | jq --raw-output '.Instances[0].InstanceId' )
	echo $INSTANCE_ID

	echo -n "Getting ip address... "
	IP=$( aws ec2 describe-instances --instance-ids "$INSTANCE_ID" | jq --raw-output '.Reservations[0].Instances[0].PublicIpAddress' )
	echo "$IP"	

fi

echo "Waiting for server to become available..."
while ! ping -c1 $IP &>/dev/null; do sleep 5; done

# set up repo and files

ssh -i ~/Downloads/thtestbed.pem ubuntu@$IP "
	mkdir ~/weights 
	aws s3 sync ~/weights s3://model-checkpoints 
	mkdir ~/data 
	aws s3 sync ~/data s3://pong-rnn-model-data
	git clone git@github.com:dmgottlieb/pong_rnn.git
	"

# Do a job!

ssh -i ~/Downloads/thtestbed.pem ubuntu@$IP "cd ~/pong-rnn; THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python do_job.py"