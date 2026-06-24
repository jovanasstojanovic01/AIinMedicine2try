import { Routes } from '@angular/router';
import { Dashboard } from './components/dashboard/dashboard';
import { PatientDetail } from './components/patient-detail/patient-detail';
import { ExaminationForm } from './components/examination-form/examination-form';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: Dashboard },
  { path: 'patient/:id', component: PatientDetail }, // :id je jedinstveni broj pacijenta
  { path: 'new-examination', component: ExaminationForm },
  { path: '**', redirectTo: 'dashboard' }, // Ako lekar ukuca pogrešan URL, vraća ga na početak
];
