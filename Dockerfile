FROM dailyco/pipecat-base:latest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
