import { AfterContentInit, ChangeDetectorRef, Component, OnInit, Signal } from '@angular/core';
import { ChartData, ChartService } from '../../services/chart.service';
import Chart from 'chart.js/auto';
import { Guid } from 'guid-typescript';
import { toSignal } from '@angular/core/rxjs-interop';
import {MatCardModule} from '@angular/material/card';

@Component({
  selector: 'app-chart',
  imports: [MatCardModule],
  templateUrl: './chart.component.html',
  styleUrl: './chart.component.scss'
})
export class ChartComponent implements AfterContentInit {

  constructor(private chartService: ChartService, private cd: ChangeDetectorRef){
    this.data = toSignal(this.chartService.getData());
  }

    id: string = Guid.create().toString()

    public chart: Chart | null = null;
  
    data: Signal<ChartData|undefined>;

    ngAfterContentInit(): void {

      this.cd.detectChanges()

      if (!this.data()){
        return;
      }

      this.chart = new Chart(this.id, {
        type: 'line',
        data: {
          labels: this.data()!.y,
          datasets: [
            {
              label: this.data()!.x2_label,
              data: this.data()!.x2,
              borderColor: 'rgb(209,134,0)',
              backgroundColor: 'rgb(209,134,0,1)',
              pointStyle: false,
            },
            {
              label: this.data()!.x1_label,
              data: this.data()!.x1,
              borderColor: 'rgb(54, 162, 235, 0.5)',
              backgroundColor: 'rgb(54, 162, 235, 0.5)',
              pointStyle: false,
            }
          ]
        },
        options: {
          animation: false,
          //aspectRatio: 2.5,
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(0, 0, 0, 0.1)',
              }
            },
            x: {
              grid: {
                color: 'rgba(0, 0, 0, 0.1)',
              },
              
            }
          },
          plugins: {
            tooltip: {
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              bodyFont: {
                size: 10,
              },
              titleFont: {
                size: 16,
                weight: 'bold',
              }
            },
            legend: {
              labels: {
                font: {
                  size: 14,
                }
              }
            }
          ,
          }
        }
      });
    }

}
