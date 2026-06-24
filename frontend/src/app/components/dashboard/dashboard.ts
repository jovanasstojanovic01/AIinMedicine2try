import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
// Uvozimo Material elemente
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';

// Definišemo strukturu podataka za tabelu
export interface ExaminationRecord {
  pacijentId: string; // Promenjeno u string radi lakšeg mapiranja ruterom
  datum: string;
  imePrezime: string;
  jmbg: string;
  tipPregleda: 'Inicijalni' | 'Kontrola';
  status: string;

  // Novi multi-modalni parametri specifični za GRAPE dataset:
  godinaPracenja: string; // Npr. 'Početno stanje', 'Godina 1', 'Godina 3'

  // Strukturni parametri (Fundus slike / Optički disk)
  cdrOD: number; // Cup-to-Disk ratio za desno oko (Oculus Dexter)
  cdrOS: number; // Cup-to-Disk ratio za levo oko (Oculus Sinister)

  // Klinički parametri
  iopOD: number; // Intraokularni pritisak desno oko (mmHg)
  iopOS: number; // Intraokularni pritisak levo oko (mmHg)

  // Funkcionalni parametri (Vidno polje / Perimetrija)
  mdOD: number; // Mean Deviation za desno oko (u decibelima - dB)
  mdOS: number; // Mean Deviation za levo oko (u decibelima - dB)
}

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
  // Nazivi kolona koje će se prikazati u tabeli
  displayedColumns: string[] = ['datum', 'imePrezime', 'jmbg', 'tipPregleda', 'status', 'akcija'];

  // Lažni podaci da lekar vidi istoriju svih pregleda zajedno
  allExaminations: ExaminationRecord[] = [
    {
      pacijentId: '1',
      datum: '31.05.2026.',
      imePrezime: 'Marko Nikolić',
      jmbg: '1508985710023',
      tipPregleda: 'Kontrola',
      status: 'Glaucoma',
      // GRAPE Specifični parametri za ovaj pregled:
      godinaPracenja: 'Godina 3',
      iopOD: 22, // Povišen pritisak desno oko
      iopOS: 16, // Normalan pritisak levo oko
      mdOD: -6.42, // Umereno oštećenje vidnog polja u dB (Mean Deviation)
      mdOS: -1.2, // Normalno/blago odstupanje vidnog polja
      cdrOD: 0.65, // Visok Cup-to-Disk ratio (Sumnja na glaukom)
      cdrOS: 0.4, // Normalan CDR odnos
    },
    {
      pacijentId: '2',
      datum: '30.05.2026.',
      imePrezime: 'Jelena Petrović',
      jmbg: '2211991715234',
      tipPregleda: 'Inicijalni',
      status: 'Negative',
      godinaPracenja: 'Početno stanje',
      iopOD: 26, // Jako visok pritisak!
      iopOS: 24,
      mdOD: -12.15, // Napredovalo oštećenje vidnog polja
      mdOS: -9.8,
      cdrOD: 0.75, // Velika ekskavacija optičkog diska
      cdrOS: 0.7,
    },
    {
      pacijentId: '3',
      datum: '28.05.2026.',
      imePrezime: 'Nikola Aleksić',
      jmbg: '0402978710055',
      tipPregleda: 'Kontrola',
      status: 'Glaucoma',
      godinaPracenja: 'Godina 1',
      iopOD: 18,
      iopOS: 17,
      mdOD: -2.1,
      mdOS: -1.95,
      cdrOD: 0.45,
      cdrOS: 0.42,
    },
  ];

  filteredExaminations: any[] = [];

  // 4. Polje za pretragu sa "getter-om" i "setter-om" koji automatski filtriraju tabelu čim kucaš
  private _searchQuery: string = '';

  get searchQuery(): string {
    return this._searchQuery;
  }

  set searchQuery(value: string) {
    this._searchQuery = value;
    this.filtrirajPacijente();
  }

  ngOnInit(): void {
    // Čim se komponenta učita, napuni tabelu svim pacijentima
    this.filteredExaminations = [...this.allExaminations];
  }

  // Funkcija koja radi pretragu po imenu ili JMBG-u
  filtrirajPacijente(): void {
    if (!this.searchQuery.trim()) {
      this.filteredExaminations = [...this.allExaminations];
      return;
    }

    const query = this.searchQuery.toLowerCase().trim();

    this.filteredExaminations = this.allExaminations.filter(
      (exam) => exam.imePrezime.toLowerCase().includes(query) || exam.jmbg.includes(query),
    );
  }
}
