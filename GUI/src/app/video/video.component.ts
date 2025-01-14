import { Component, ElementRef, ViewChild } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import {MatSelectChange, MatSelectModule} from '@angular/material/select';
import {FormsModule} from '@angular/forms';
import {MatInputModule} from '@angular/material/input';
import {MatFormFieldModule} from '@angular/material/form-field';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-video',
  imports: [MatCardModule, MatFormFieldModule, MatSelectModule, MatInputModule, FormsModule, CommonModule],
  templateUrl: './video.component.html',
  styleUrl: './video.component.scss'
})
export class VideoComponent {
  videos: string[] = [
    '17:30-episode-0.mp4',
    '17:31-episode-0.mp4',
    '17:33-episode-0.mp4'
  ];

  selected = this.videos[0];

  @ViewChild('video', { static: false }) video!: ElementRef;

  selectionChange(event: MatSelectChange){
    this.video.nativeElement.src = event.source.value
  }
}
