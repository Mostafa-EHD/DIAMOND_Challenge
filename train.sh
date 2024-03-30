# First run with a 3-day limit. Convert days to hours: 3 days * 24 hours/day = 72 hours
timeout 72h python main.py --learning_rate 1e-4 --batch_size 16 --backbone resnet50

# Second run with a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours
timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone efficientnet_b0

# Third run with a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours
timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone resnet101