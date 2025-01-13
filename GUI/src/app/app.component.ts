import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { ChartComponent } from './chart/chart.component';
import { MatCardModule } from '@angular/material/card';
import { VideoComponent } from "./video/video.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, MatButtonModule, ChartComponent, MatCardModule, VideoComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent{

  constructor(){

  }

  

}
