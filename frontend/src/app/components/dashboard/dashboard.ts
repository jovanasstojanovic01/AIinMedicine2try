import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';

// Uvoz servisa i modela
import { PatientService } from '../../core/http/patient.service';
import { Patient } from '../../core/models/patient.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    MatTableModule,
    MatButtonModule,
    MatInputModule,
    MatFormFieldModule,
    MatIconModule,
    MatCardModule,
  ],
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.scss'],
})
export class Dashboard implements OnInit {
  // Kolone usklađene sa stvarnim Flask parametrima
  displayedColumns: string[] = [
    'imePrezime',
    'birth_date',
    'gender',
    'cct',
    'glaucoma_category',
    'akcija',
  ];

  // Lista pacijenata koja se vezuje za mat-table
  patients: Patient[] = [];

  // Model za dvosmerno vezivanje pretrage (ngModel)
  searchQuery: string = '';

  // Injektovan ChangeDetectorRef (cdr) za trenutno osvežavanje pogleda
  constructor(
    private patientService: PatientService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.ucitajPacijente();
  }

  // Glavna funkcija za povlačenje podataka sa backenda
  ucitajPacijente(): void {
    this.patientService.getAllPatients(this.searchQuery, 1).subscribe({
      next: (response: any) => {
        // Tačno mapiranje: Flask podatke pakuje u response.data.patients
        if (response && response.data && response.data.patients) {
          this.patients = response.data.patients;
        } else {
          this.patients = [];
        }

        // Forsiramo Angular da odmah detektuje izmenu niza i iscrta tabelu na ekranu
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Greška pri učitavanju pacijenata sa backenda:', err);
      },
    });
  }

  // Pokreće se kada lekar ukuca pojam ili klikne na ikonicu lupe
  onSearch(): void {
    this.ucitajPacijente();
  }
}
