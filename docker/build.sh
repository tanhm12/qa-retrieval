branch=$(git branch --show-current)
commit_head=$(git log -1 --format="%H")
echo $branch > repo_info.txt
echo $commit_head >> repo_info.txt

set -e

PASSPHRASE=tanhm

IMAGE_VERSION=$(date "+%Y%m%d")
commit_head=$(echo $commit_head | sed -e "s/^\(.\{8\}\).*/\1/" )
docker build -t tanhm/qa-retrieval:latest -f Dockerfile \
    --build-arg passphrase=${PASSPHRASE} \
    --build-arg CACHEBUST=$(date +%s) \
    --build-arg commit_head=$commit_head --build-arg branch=$branch \
    . 2>&1 