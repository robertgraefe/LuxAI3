import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export type ChartData = {
  x1_label: string,
  x2_label: string,
  y_label: string,
  x1: Array<number>,
  x2: Array<number|null>,
  y: Array<number>
}

@Injectable({
  providedIn: 'root'
})
export class ChartService {

  constructor(private http: HttpClient) { 
    
  }

  getData(): Observable<ChartData>{
    return this.http.get<ChartData>("data.json");
    //return this.http.get<ChartData>("data2.json");
  }
}
