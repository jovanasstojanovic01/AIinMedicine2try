import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatTabsModule } from '@angular/material/tabs';

import { PatientService } from '../../core/http/patient.service';
import { VisitService } from '../../core/http/visit.service';

@Component({
  selector: 'app-patient-detail',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatListModule,
    MatTabsModule,
  ],
  templateUrl: './patient-detail.html',
  styleUrls: ['./patient-detail.scss'],
})
export class PatientDetail implements OnInit {
  patientId!: number;
  pacijent: any = null;
  pregledi: any[] = [];
  izabraniPregled: any = null;
  prikaziMasku: boolean = false;

  mediaUrl = 'http://127.0.0.1:5000/api/media';

  constructor(
    private route: ActivatedRoute,
    private patientService: PatientService,
    private visitService: VisitService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.patientId = Number(this.route.snapshot.paramMap.get('id'));

    if (this.patientId) {
      this.ucitajPodatkePacijenta();
      this.ucitajPregledePacijenta();
    }
  }

  ucitajPodatkePacijenta(): void {
    this.patientService.getPatientById(this.patientId).subscribe({
      next: (response: any) => {
        if (response && response.data) {
          this.pacijent = response.data;
        } else {
          this.pacijent = response;
        }
        this.cdr.detectChanges();
      },
      error: (err) => console.error('Greška pri učitavanju pacijenta:', err),
    });
  }

  ucitajPregledePacijenta(): void {
    this.visitService.getExamsByPatient(this.patientId).subscribe({
      next: (response: any) => {
        if (response && response.data && response.data.visits) {
          this.pregledi = response.data.visits;
        } else if (response && response.data) {
          this.pregledi = Array.isArray(response.data) ? response.data : [];
        } else {
          this.pregledi = Array.isArray(response) ? response : [];
        }

        if (this.pregledi.length > 0) {
          this.izabraniPregled = this.pregledi[0];
        }
        this.cdr.detectChanges();
      },
      error: (err) => console.error('Greška pri učitavanju pregleda:', err),
    });
  }

  selektujPregled(pregled: any): void {
    this.izabraniPregled = pregled;
    this.cdr.detectChanges();
  }

  toggleMaska(): void {
    this.prikaziMasku = !this.prikaziMasku;
    this.cdr.detectChanges();
  }
}
