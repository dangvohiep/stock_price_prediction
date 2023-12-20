from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import pandas as pd
from main import predict

@api_view(['GET'])
def model_api(request):
    symbol = request.GET.get('symbol', '')
    backward = int(request.GET.get('backward', '100'))
    forward = int(request.GET.get('forward', '5'))
    if not symbol:
        return JsonResponse({'error': 'No symbol provided'}, status=400)
    return JsonResponse(predict(symbol, backward, forward))

